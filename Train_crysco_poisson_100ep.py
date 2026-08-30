"""
CrysCo Retraining Script — Poisson Ratio, 100 Epochs
Fixed test/val/train split by structure_id, matching CGCNN's test set exactly.

Usage:
    python train_crysco_poisson_100ep.py
(edit the CONFIG block below for your paths)
"""

import os
import csv
import numpy as np
import torch
from torch_geometric.data import DataLoader

from crysco.models.CrysCo import CrysCo
from crysco.utils.utils_train import (
    train_model, model_setup, evaluate, write_results
)
from crysco.data.data import StructureDataset, GetY

# ----------------------------- CONFIG -----------------------------
DATA_DIR      = r"D:\MyProjects\crysco_benchmark"          # contains processed/
PROCESSED_FILE = "poisson_ratio.pt"                                 # <-- set to your actual .pt filename in processed/
SPLIT_DIR     = r"D:\MyProjects\crysco_benchmark\split\poisson_ratio" 
TRAIN_IDS_CSV = os.path.join(SPLIT_DIR, "crysco_train_ids.csv")
VAL_IDS_CSV   = os.path.join(SPLIT_DIR, "crysco_val_ids.csv")
TEST_IDS_CSV  = os.path.join(SPLIT_DIR, "crysco_test_ids.csv")   # exact CGCNN test set

OUT_DIR       = DATA_DIR
JOB_NAME      = "crysco_poisson_100ep"
MODEL_SAVE_PATH = os.path.join(OUT_DIR, f"{JOB_NAME}_best.pth")

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
SEED     = 42

model_parameters = {
    "out_dims": 64,
    "d_model": 512,
    "N": 3,
    "heads": 4,
    "dim1": 64,
    "dim2": 150,
    "numb_embbeding": 1,
    "numb_EGAT": 5,
    "numb_GATGCN": 1,
    "pool": "global_add_pool",
    "pool_order": "early",
    "act": "silu",
    "model": "CrysCo",
    "dropout_rate": 0.0,
    "epochs": 100,
    "lr": 0.001,
    "batch_size": 32,
    "optimizer": "AdamW",
    "optimizer_args": {},
    "scheduler": "ReduceLROnPlateau",
    "scheduler_args": {
        "mode": "min", "factor": 0.8, "patience": 15,
        "min_lr": 0.00001, "threshold": 0.0002,
    },
}
LOSS_FN   = "mse_loss"
VERBOSITY = 1
# --------------------------------------------------------------


def load_id_list(csv_path):
    """Read a single-column (or first-column) CSV of material IDs into a set of strings."""
    ids = set()
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            val = row[0].strip()
            if val.lower() in ("id", "material_id", "structure_id"):
                continue  # skip header if present
            ids.add(val)
    return ids


def get_structure_id_str(data_item):
    """CrysCo stores structure_id as a nested list, e.g. [['mp-1234']]. Flatten to a plain string."""
    sid = data_item.structure_id
    while isinstance(sid, (list, tuple)):
        sid = sid[0]
    return str(sid)


def split_by_fixed_ids(dataset, train_ids, val_ids, test_ids):
    """Partition dataset into train/val/test Subsets using exact ID membership."""
    train_idx, val_idx, test_idx = [], [], []
    missing = []

    for i in range(len(dataset)):
        sid = get_structure_id_str(dataset[i])
        if sid in test_ids:
            test_idx.append(i)
        elif sid in val_ids:
            val_idx.append(i)
        elif sid in train_ids:
            train_idx.append(i)
        else:
            missing.append(sid)

    print(f"Matched -> train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")
    if missing:
        print(f"WARNING: {len(missing)} structures in dataset not found in any split CSV "
              f"(first 5: {missing[:5]})")

    expected_test = len(test_ids)
    if len(test_idx) != expected_test:
        print(f"WARNING: expected {expected_test} test structures from CSV, "
              f"matched only {len(test_idx)} in dataset. Check ID formatting "
              f"(e.g. 'mp-1234' vs 'mp-1234.cif' vs int).")

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)
    test_dataset = torch.utils.data.Subset(dataset, test_idx)
    return train_dataset, val_dataset, test_dataset


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print(f"Using device: {DEVICE}")
    print(f"Training config: epochs={model_parameters['epochs']}, "
          f"batch_size={model_parameters['batch_size']}, lr={model_parameters['lr']}")

    # ---- Load full processed dataset ----
    transforms = GetY(index=0)
    dataset = StructureDataset(
        DATA_DIR,
        processed_path="processed",
        filename=PROCESSED_FILE,
        transform=transforms,
    )
    print(f"Full dataset size: {len(dataset)}")

    # ---- Load exact CGCNN-matching split ----
    train_ids = load_id_list(TRAIN_IDS_CSV)
    val_ids = load_id_list(VAL_IDS_CSV)
    test_ids = load_id_list(TEST_IDS_CSV)
    print(f"Split CSV sizes -> train: {len(train_ids)}, val: {len(val_ids)}, "
          f"test: {len(test_ids)} (test set must exactly match CGCNN's test_results.csv IDs)")

    train_dataset, val_dataset, test_dataset = split_by_fixed_ids(
        dataset, train_ids, val_ids, test_ids
    )

    # ---- Data loaders ----
    train_loader = DataLoader(
        train_dataset, batch_size=model_parameters["batch_size"],
        shuffle=True, num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=model_parameters["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True,
    ) if len(val_dataset) > 0 else None
    test_loader = DataLoader(
        test_dataset, batch_size=model_parameters["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True,
    ) if len(test_dataset) > 0 else None

    # ---- Model / optimizer / scheduler ----
    model = model_setup(DEVICE, "CrysCo", model_parameters, dataset)

    optimizer = getattr(torch.optim, model_parameters["optimizer"])(
        model.parameters(), lr=model_parameters["lr"], **model_parameters["optimizer_args"]
    )
    scheduler = getattr(torch.optim.lr_scheduler, model_parameters["scheduler"])(
        optimizer, **model_parameters["scheduler_args"]
    )

    # ---- Train (saves best-val checkpoint to MODEL_SAVE_PATH automatically) ----
    model = train_model(
        DEVICE, 0, model, optimizer, scheduler, LOSS_FN,
        train_loader, val_loader, None,
        model_parameters["epochs"], VERBOSITY,
        MODEL_SAVE_PATH,
    )

    # ---- Final evaluation with best model, in eval mode, no shuffle ----
    eval_train_loader = DataLoader(
        train_dataset, batch_size=model_parameters["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True,
    )

    train_error, train_out = evaluate(eval_train_loader, model, LOSS_FN, DEVICE, out=True)
    print(f"Train Error (MSE): {train_error:.5f}")

    if val_loader is not None:
        val_error, val_out = evaluate(val_loader, model, LOSS_FN, DEVICE, out=True)
        print(f"Val Error (MSE): {val_error:.5f}")
        write_results(val_out, os.path.join(OUT_DIR, f"{JOB_NAME}_val_outputs.csv"))

    if test_loader is not None:
        test_error, test_out = evaluate(test_loader, model, LOSS_FN, DEVICE, out=True)
        print(f"Test Error (MSE): {test_error:.5f}")
        write_results(test_out, os.path.join(OUT_DIR, f"{JOB_NAME}_test_outputs.csv"))

    write_results(train_out, os.path.join(OUT_DIR, f"{JOB_NAME}_train_outputs.csv"))

    print(f"Best checkpoint saved to: {MODEL_SAVE_PATH}")
    print(f"Outputs saved with prefix: {JOB_NAME}_*.csv in {OUT_DIR}")


if __name__ == "__main__":
    main()