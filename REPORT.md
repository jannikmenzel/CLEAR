# CLEAR Experiment Report: ODIR-OIA Dataset

## 1. Dataset

The **ODIR-OIA** dataset is based on the Ocular Disease Intelligent Recognition (ODIR-2019) challenge [1], using the version processed by Li et al. [2]. It comprises fundus photographs and clinical keywords from both eyes of **5,000 patients**, sourced from multiple hospitals and devices across China. Each patient is annotated with binary labels for **8 disease categories** (multi-label): Normal (N), Diabetes (D), Glaucoma (G), Cataract (C), Age-related Macular Degeneration (A), Hypertension (H), Myopia (M), and Other (O). Three input modalities are available per patient: demographics (age + sex), fundus images (left + right), and diagnostic keyword text (left + right).

The dataset provides pre-defined splits:

| Split          | Patients | Source Set        |
|----------------|----------|-------------------|
| **Training**   | 3,500    | Training Set      |
| **Validation** | 500      | Off-site Test Set |
| **Test**       | 1,000    | On-site Test Set  |
| **Total**      | 5,000    |                   |

[1] ODIR-2019 Challenge. https://odir2019.grand-challenge.org

[2] Li, N., Li, T., Hu, C., Wang, K., & Kang, H. (2021). A Benchmark of Ocular Disease Intelligent Recognition: One Shot for Multi-disease Detection. *arXiv:2102.07978*.

## 2. Setup Comparison

| Component           | Paper Setup                         | This Implementation                   |
|---------------------|-------------------------------------|---------------------------------------|
| Hardware            | 3× NVIDIA RTX 2080 Ti (33 GB total) | Auto-detected (MPS/CUDA/CPU)          |
| Epochs              | 50                                  | 50                                    |
| Learning Rate       | 5e-4                                | 5e-4                                  |
| Optimizer           | Adam                                | Adam                                  |
| Weight Decay        | 1e-5                                | 1e-5                                  |
| Batch Size          | 16                                  | 16                                    |
| LR Scheduler        | Adaptive decay (factor 0.3)         | ReduceLROnPlateau (factor 0.3)        |
| Image Encoder       | ResNet18 (pretrained, frozen)       | ResNet18 (pretrained, frozen)         |
| Image Feature Dim   | 768 per eye                         | 768 per eye (projected via nn.Linear) |
| Text Encoder        | ClinicalBERT (frozen)               | ClinicalBERT (frozen)                 |
| Keyword Feature Dim | 768 per eye                         | 768 per eye                           |
| Embed Dim           | 128                                 | 128                                   |
| Num Heads           | 8                                   | 8                                     |
| Layers (L)          | 2                                   | 2                                     |
| Cross Layers        | 2                                   | 2                                     |
| Counterfactual Loss | Enabled                             | Enabled                               |
| Evaluation Split    | Test                                | Test                                  |
| Runs                | 3 (average)                         | 3 (Seeds: 42, 123, 456)               |

### Modalities

| # | Modality          | Dim  | Description                                                         |
|---|-------------------|------|---------------------------------------------------------------------|
| 1 | Demographics      | 2    | age (normalized) + sex (binary)                                     |
| 2 | Fundus images     | 1536 | left + right ResNet18 embeddings (768 each), concatenated           |
| 3 | Clinical keywords | 1536 | left + right ClinicalBERT [CLS] embeddings (768 each), concatenated |

## 4. Results

### Result Comparison

| Label         | Paper (AUC) | Ours (AUC)       |
|---------------|-------------|------------------|
| N             | 0.9998      | 0.9999±.0000     |
| D             | 0.9991      | 0.9994±.0003     |
| G             | 0.9933      | 0.9956±.0024     |
| C             | 0.9999      | 0.9975±.0013     |
| A             | 0.9912      | 0.9989±.0001     |
| H             | 0.9948      | 0.9983±.0006     |
| M             | 0.9977      | 0.9830±.0150     |
| O             | 0.9969      | 0.9977±.0004     |
| **Macro-AUC** | **0.9969**  | **0.9963±.0018** |
| **Micro-AUC** | **0.9986**  | **0.9981±.0007** |
| **Micro-F1**  | —           | **0.9556±.0032** |

## 3. Usage

```bash
# 1. Prepare data
python data/prepare_odir_paper.py --odir_oia_root ./data/ODIR-OIA --output_dir ./data/odir_oia

# 2. Train 3 models
python run_experiments.py

# 3. Evaluate all runs and compute mean ± std
python eval_runs.py
```
