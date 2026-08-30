"""
Get chemical formula + composition for the 30 BEST and 30 WORST CrysCo test
predictions, so material type / class can be identified and mapped to
industrial applications.

Requires: pip install pymatgen --break-system-packages   (if not installed)

Run:
    python get_formulas_best_worst.py
"""

import os
import pandas as pd
from pymatgen.core import Structure

# ----------------------------- CONFIG -----------------------------
TEST_OUTPUTS_CSV = r"D:\MyProjects\crysco_benchmark\results\crysco_poisson_100ep_test_outputs.csv"
CIF_DIR = r"D:\MyProjects\crysco_benchmark\structures\poisson_ratio_final"
TOP_N = 30
OUT_CSV = "best_worst_with_formulas.csv"
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
    for name in (f"{mp_id}.cif", f"{mp_id}.CIF"):
        p = os.path.join(cif_dir, name)
        if os.path.exists(p):
            return p
    return None


def get_formula_info(cif_path):
    try:
        s = Structure.from_file(cif_path)
        formula = s.composition.reduced_formula
        elements = sorted(set(el.symbol for el in s.composition.elements))
        n_atoms = s.num_sites
        spacegroup = s.get_space_group_info()[0]
        return formula, ",".join(elements), n_atoms, spacegroup
    except Exception as e:
        return f"ERROR: {e}", "", "", ""


def main():
    df = load_predictions(TEST_OUTPUTS_CSV)
    df["target"] = df["target"].astype(float)
    df["prediction"] = df["prediction"].astype(float)
    df["abs_error"] = (df["target"] - df["prediction"]).abs()
    df_sorted = df.sort_values("abs_error", ascending=False).reset_index(drop=True)

    worst = df_sorted.head(TOP_N).copy()
    worst["group"] = "WORST"
    best = df_sorted.tail(TOP_N).copy()
    best["group"] = "BEST"

    combined = pd.concat([worst, best], ignore_index=True)

    formulas, elements_list, n_atoms_list, sg_list = [], [], [], []
    for mp_id in combined["id"]:
        cif_path = find_cif_path(mp_id, CIF_DIR)
        if cif_path is None:
            formulas.append("CIF NOT FOUND")
            elements_list.append("")
            n_atoms_list.append("")
            sg_list.append("")
            continue
        formula, elements, n_atoms, sg = get_formula_info(cif_path)
        formulas.append(formula)
        elements_list.append(elements)
        n_atoms_list.append(n_atoms)
        sg_list.append(sg)

    combined["formula"] = formulas
    combined["elements"] = elements_list
    combined["n_atoms"] = n_atoms_list
    combined["spacegroup"] = sg_list

    cols = ["group", "id", "target", "prediction", "abs_error",
            "formula", "elements", "n_atoms", "spacegroup"]
    combined = combined[cols]

    print("=== WORST 30 ===")
    print(combined[combined["group"] == "WORST"].to_string(index=False))
    print()
    print("=== BEST 30 ===")
    print(combined[combined["group"] == "BEST"].to_string(index=False))

    combined.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to: {OUT_CSV}")


if __name__ == "__main__":
    main()