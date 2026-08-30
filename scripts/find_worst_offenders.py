"""
Rank ALL CrysCo test predictions by absolute error (not just the known 17),
so you can pull the actual .cif files of the worst offenders and inspect them
by hand (check for malformed structure, weird lattice params, disordered
sites, etc.)

Run:
    python find_worst_offenders.py
"""

import os
import pandas as pd

# ----------------------------- CONFIG -----------------------------
TEST_OUTPUTS_CSV = r"D:\MyProjects\crysco_benchmark\results\crysco_poisson_100ep_test_outputs.csv"
CIF_DIR = r"D:\MyProjects\crysco_benchmark\structures\poisson_ratio"  # adjust if different
TOP_N = 30
OUT_CSV = "worst_offenders_ranked.csv"
# --------------------------------------------------------------


def load_predictions(csv_path):
    with open(csv_path) as f:
        first_line = f.readline()
    if "target" in first_line.lower():
        df = pd.read_csv(csv_path)
        df.columns = [c.strip().lower() for c in df.columns]
        df = df.rename(columns={"ids": "id"})
    else:
        df = pd.read_csv(csv_path, header=None, names=["id", "target", "prediction"])
    df["id"] = df["id"].astype(str).str.strip()
    return df


def find_cif_path(mp_id, cif_dir):
    # try common naming patterns
    candidates = [
        os.path.join(cif_dir, f"{mp_id}.cif"),
        os.path.join(cif_dir, f"{mp_id}.CIF"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "NOT FOUND (check naming pattern in your structures folder)"


def main():
    df = load_predictions(TEST_OUTPUTS_CSV)
    df["target"] = df["target"].astype(float)
    df["prediction"] = df["prediction"].astype(float)
    df["abs_error"] = (df["target"] - df["prediction"]).abs()

    df_sorted = df.sort_values("abs_error", ascending=False).reset_index(drop=True)
    top = df_sorted.head(TOP_N).copy()
    top["cif_path"] = top["id"].apply(lambda x: find_cif_path(x, CIF_DIR))

    print(f"Total test predictions: {len(df)}")
    print(f"Mean abs error: {df['abs_error'].mean():.5f}")
    print(f"Median abs error: {df['abs_error'].median():.5f}")
    print()
    print(f"Top {TOP_N} worst offenders:")
    print(top[["id", "target", "prediction", "abs_error", "cif_path"]].to_string(index=False))

    df_sorted.to_csv(OUT_CSV, index=False)
    print(f"\nFull ranked list ({len(df_sorted)} rows) saved to: {OUT_CSV}")
    print("Open cif_path entries above in a structure viewer (VESTA / pymatgen) "
          "to check for malformed cells, disorder, huge lattice params, etc.")


if __name__ == "__main__":
    main()