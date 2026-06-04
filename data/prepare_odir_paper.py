import argparse
import os
import pickle
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ODIR-OIA dataset with paper-correct 3-modality setup")
    parser.add_argument("--odir_oia_root", type=str, required=True,
                        help="Path to ODIR-OIA root. Expects Training Set/, Off-site Test Set/, On-site Test Set/ inside.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_encoder", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--max_samples", type=int, default=0, help="Limit number of patients for testing")
    return parser.parse_args()


class ImageEncoder:
    def __init__(self, model_name: str = 'resnet18', device: str = None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.embedding_dim = 2048
        elif model_name == 'resnet18':
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
            self.embedding_dim = 512
        self.model = self.model.to(self.device)
        self.model.eval()
        # Project to 768 per eye (paper's modality feature dimension)
        self.proj = nn.Linear(self.embedding_dim, 768).to(self.device)
        self.proj.eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def encode(self, image_path: str) -> np.ndarray:
        try:
            img = Image.open(image_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(img_tensor).squeeze()
                if self.proj is not None:
                    embedding = self.proj(embedding)
                return embedding.cpu().numpy()
        except Exception as e:
            print(f"Error encoding {image_path}: {e}")
            return np.zeros(768)


class TextEncoder:
    def __init__(self, model_name: str = 'ClinicalBERT', device: str = None):
        self.device = device or ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained("medicalai/ClinicalBERT")
        self.model = AutoModel.from_pretrained("medicalai/ClinicalBERT")
        self.model = self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.embedding_dim = 768

    def encode(self, text: str) -> np.ndarray:
        if not text or text == 'None':
            return np.zeros(self.embedding_dim)
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return embedding


def parse_odir_xlsx(xlsx_path: str) -> List[Dict]:
    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    ws = wb.active
    header = next(ws.iter_rows(values_only=True))
    header = [str(h).strip() if h is not None else f"col_{i}" for i, h in enumerate(header)]

    rows = list(ws.iter_rows(values_only=True))
    rows = [r for r in rows if r[0] != header[0]]  # skip duplicate header row

    data = []
    for row in tqdm(rows, desc="Parsing Excel"):
        row_data = {}
        for i in range(min(len(header), len(row))):
            row_data[header[i]] = row[i]
        data.append(row_data)

    wb.close()
    return data


def find_image(set_dir: str, filename: str) -> str | None:
    path = os.path.join(set_dir, "Images", filename)
    if os.path.exists(path):
        return path
    return None


def load_splits(odir_oia_root: str, max_samples: int = 0):
    sets = [
        ("train", "Training Set", "training annotation (English).xlsx"),
        ("val", "Off-site Test Set", "off-site test annotation (English).xlsx"),
        ("test", "On-site Test Set", "on-site test annotation (English).xlsx"),
    ]
    splits = []
    for split_name, set_folder, anno_file in sets:
        anno_path = os.path.join(odir_oia_root, set_folder, "Annotation", anno_file)
        data = parse_odir_xlsx(anno_path)
        if max_samples > 0:
            data = data[:max_samples]
        set_dir = os.path.join(odir_oia_root, set_folder)
        splits.append((split_name, data, set_dir))
        print(f"  {split_name}: {len(data)} patients from {set_folder}")
    return splits


def main():
    args = parse_args()
    print(f"Loading ODIR-OIA from {args.odir_oia_root}")
    splits = load_splits(args.odir_oia_root, args.max_samples)

    print(f"Initializing {args.image_encoder} image encoder...")
    image_encoder = ImageEncoder(args.image_encoder)
    img_embed_dim = image_encoder.embedding_dim
    img_trunc_dim = 768

    print("Initializing ClinicalBERT text encoder for diagnostic keywords...")
    text_encoder = TextEncoder()

    os.makedirs(args.output_dir, exist_ok=True)

    for split_name, split_data, set_dir in splits:
        print(f"Processing {split_name} set...")
        processed = []

        for patient in tqdm(split_data, desc=f"Processing {split_name}"):
            try:
                # 1. Demographics [2]: age (normalized) + sex (binary)
                age = float(patient['Patient Age']) / 100.0 if patient['Patient Age'] is not None else 0.5
                sex = 1.0 if str(patient['Patient Sex']).lower() == 'male' else 0.0
                demographics = np.array([age, sex], dtype=np.float32)

                # 2. Fundus images L+R concatenated [1536]
                left_img = str(patient['Left-Fundus'])
                right_img = str(patient['Right-Fundus'])
                left_path = find_image(set_dir, left_img)
                right_path = find_image(set_dir, right_img)

                left_emb = image_encoder.encode(left_path) if left_path else np.zeros(img_embed_dim)
                right_emb = image_encoder.encode(right_path) if right_path else np.zeros(img_embed_dim)

                # Truncate to 768 per eye (paper uses 768-dim image embeddings)
                if len(left_emb.shape) == 0:
                    left_emb = np.zeros(img_trunc_dim)
                if len(right_emb.shape) == 0:
                    right_emb = np.zeros(img_trunc_dim)
                left_emb = left_emb[:img_trunc_dim]
                right_emb = right_emb[:img_trunc_dim]
                fundus = np.concatenate([left_emb, right_emb]).astype(np.float32)  # (1536,)

                # 3. Clinical keywords L+R concatenated [1536] via frozen ClinicalBERT
                kw_l = str(patient.get('Left-Diagnostic Keywords', ''))
                kw_r = str(patient.get('Right-Diagnostic Keywords', ''))
                kw_l_emb = text_encoder.encode(kw_l)
                kw_r_emb = text_encoder.encode(kw_r)
                keywords = np.concatenate([kw_l_emb, kw_r_emb]).astype(np.float32)  # (1536,)

                # 4. Labels [8]
                label_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
                labels = np.array([int(patient.get(n, 0) == 1) for n in label_names], dtype=np.float32)

                processed.append({
                    'stay_id': str(patient['ID']),
                    'demographics': demographics,
                    'fundus': fundus,
                    'keywords': keywords,
                    'labels': labels,
                })
            except Exception as e:
                print(f"Error processing patient {patient.get('ID')}: {e}")
                continue

        output_path = os.path.join(args.output_dir, f"{split_name}.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(processed, f)
        print(f"Saved {len(processed)} samples to {output_path}")

    # Print summary
    print("\n=== Summary ===")
    for split_name, split_data, _ in splits:
        labels = np.array([[int(d.get(n, 0) == 1) for n in ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']] for d in split_data])
        total_pos = labels.sum()
        total_entries = labels.size
        print(f"{split_name.upper()}: {len(split_data)} patients, positive rate: {total_pos/total_entries*100:.2f}%")
        label_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
        for i, name in enumerate(label_names):
            print(f"  {name}: {labels[:, i].sum()} positive ({labels[:, i].mean()*100:.2f}%)")


if __name__ == '__main__':
    main()
