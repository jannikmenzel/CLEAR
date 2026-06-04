import subprocess
import sys
import os

SEEDS = [42, 123, 456]
DATA_PATH = "./data/odir_oia/"
OUTPUT_TEMPLATE = "./weights/clear_run_{seed}"
TASK = "odir_paper"
NUM_EPOCHS = 50

def train():
    for seed in SEEDS:
        output_path = OUTPUT_TEMPLATE.format(seed=seed)
        ckpt_path = output_path + "_aupr_0"
        if os.path.exists(ckpt_path):
            print(f"[SKIP] {ckpt_path} already exists.")
            continue
        print(f"\n{'='*60}")
        print(f"  Training model with seed {seed}")
        print(f"{'='*60}")
        subprocess.run([
            sys.executable, "train.py",
            "--task", TASK,
            "--data_path", DATA_PATH,
            "--seed", str(seed),
            "--output_path", output_path,
            "--num_train_epochs", str(NUM_EPOCHS),
        ], check=True)

def evaluate():
    print(f"\n{'='*60}")
    print(f"  Evaluating all {len(SEEDS)} models")
    print(f"{'='*60}")
    subprocess.run([sys.executable, "eval_runs.py"], check=True)

if __name__ == "__main__":
    train()
