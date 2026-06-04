import torch, numpy as np, pickle
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score

device = torch.device("cpu")
label_names = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']

from model.CLEAR import CLEAR

class Args:
    task = 'odir_paper'; num_heads = 8; dropout = 0.1; layers = 2
    cross_layers = 2; num_of_labtests = 12; num_of_notes = 5
    language_model = 'ClinicalBERT'
    modalities = ['category_attributes', 'numerical_sequence', 'category_sequence1', 'category_sequence2', 'notes']
    num_modalities = 5
args = Args()

embed_dim = 128; rnn_dim = 64
textembed_dim = 1536; triage_dim = 2; num_class = 8
hidden_dim = embed_dim * 5 * 2
numerical_sequence_parameters = {"input_dim": 7, "rnn_dim": rnn_dim}
category_sequence_parameters1 = {'input_dim': 1536, 'embed_dim': embed_dim, 'rnn_dim': rnn_dim}
category_sequence_parameters2 = {'input_dim': 10321, 'embed_dim': embed_dim, 'rnn_dim': rnn_dim}

task_template = torch.zeros((2, 768), dtype=torch.float32)
mask_template = torch.zeros(768, dtype=torch.float32)

test_data_path = './data/odir_oia/test.pkl'
runs = [
    (1, 42, 'clear_run_42_aupr_0'),
    (2, 123, 'clear_run_123_aupr_0'),
    (3, 456, 'clear_run_456_aupr_0'),
]

y_true_list, y_pred_list = [], []

for run, seed, ckpt_name in runs:
    ckpt = f'./weights/{ckpt_name}'
    print(f"\n=== Run {run} (seed {seed}) ===")

    with open(test_data_path, 'rb') as f:
        val_data = pickle.load(f)

    model = CLEAR(args, triage_dim, numerical_sequence_parameters, category_sequence_parameters1,
                  category_sequence_parameters2, textembed_dim, embed_dim, hidden_dim,
                  device, output_dim=num_class)
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint['net'], strict=False)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for item in tqdm(val_data):
            t = torch.FloatTensor(item['demographics']).unsqueeze(0)
            tx = torch.FloatTensor(item['fundus']).unsqueeze(0).unsqueeze(0)
            md = torch.FloatTensor(item['keywords']).unsqueeze(0).unsqueeze(0)
            lt = torch.zeros((1, 12, 7), dtype=torch.float32)
            dg = torch.zeros((1, 1, 10321), dtype=torch.float32)
            lb = torch.FloatTensor(item['labels']).unsqueeze(0)

            out = model(triage_variables=t, labtest=lt, text_representations=tx,
                        medication_one_hot_tensors=md, diagnoses_one_hot_tensors=dg,
                        task_template=task_template.unsqueeze(0),
                        mask_template=mask_template.unsqueeze(0),
                        tao=0.3, mode='val')
            y_pred.append(out.cpu().numpy()[0])
            y_true.append(lb.cpu().numpy()[0])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    macro_auc = []
    for i in range(8):
        auc = roc_auc_score(y_true[:, i], y_pred[:, i])
        macro_auc.append(auc)
        print(f"  {label_names[i]}: AUC = {auc:.4f}")

    y_pred_bin = (y_pred > 0.5).astype(int)
    micro_f1 = f1_score(y_true, y_pred_bin, average='micro')
    macro_auc_val = np.mean(macro_auc)
    micro_auc_val = roc_auc_score(y_true, y_pred, average='micro')
    print(f"  Macro-AUC: {macro_auc_val:.4f}")
    print(f"  Micro-AUC: {micro_auc_val:.4f}")
    print(f"  Micro-F1:  {micro_f1:.4f}")

    y_true_list.append(y_true)
    y_pred_list.append(y_pred)

if len(runs) > 1:
    print(f"\n{'='*50}")
    print(f"  MEAN OVER {len(runs)} RUNS")
    print(f"{'='*50}")

    all_macro_auc, all_micro_auc, all_micro_f1 = [], [], []
    all_per_label = [[] for _ in range(8)]

    for i in range(len(runs)):
        yt, yp = y_true_list[i], y_pred_list[i]
        pl = []
        for j in range(8):
            auc = roc_auc_score(yt[:, j], yp[:, j])
            pl.append(auc)
            all_per_label[j].append(auc)
        all_macro_auc.append(np.mean(pl))
        all_micro_auc.append(roc_auc_score(yt, yp, average='micro'))
        yb = (yp > 0.5).astype(int)
        all_micro_f1.append(f1_score(yt, yb, average='micro'))

    for j, name in enumerate(label_names):
        print(f"  {name}: {np.mean(all_per_label[j]):.4f} ±{np.std(all_per_label[j]):.4f}")
    print(f"  Macro-AUC: {np.mean(all_macro_auc):.4f} ±{np.std(all_macro_auc):.4f}")
    print(f"  Micro-AUC: {np.mean(all_micro_auc):.4f} ±{np.std(all_micro_auc):.4f}")
    print(f"  Micro-F1:  {np.mean(all_micro_f1):.4f} ±{np.std(all_micro_f1):.4f}")
