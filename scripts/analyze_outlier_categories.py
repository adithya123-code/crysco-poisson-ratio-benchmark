"""
Pull chemical formula, elements, atom count, space group for the 17 known
CrysCo extreme-outlier material IDs, combined with their error history
(20ep vs 100ep vs CGCNN) from outlier_diagnostic_20ep_vs_100ep.csv.

Run:
    python analyze_outlier_categories.py
"""

import os
import pandas as pd
from pymatgen.core import Structure

# ----------------------------- CONFIG -----------------------------
OUTLIER_CSV = r"D:\MyProjects\crysco_benchmark\results\outlier_diagnostic_20ep_vs_100ep.csv"
CIF_DIR = r"D:\MyProjects\crysco_benchmark\structures\poisson_ratio_final"
OUT_CSV = "outlier_17_full_analysis.csv"
# --------------------------------------------------------------


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
        density = round(s.density, 3)
        volume_per_atom = round(s.volume / s.num_sites, 3)
        return formula, ",".join(elements), n_atoms, spacegroup, density, volume_per_atom
    except Exception as e:
        return f"ERROR: {e}", "", "", "", "", ""


def classify(elements_str, n_atoms):
    """Rough heuristic categorisation for discussion purposes."""
    elements = elements_str.split(",") if elements_str else []
    n_el = len(elements)
    metals = {"Li","Na","K","Rb","Cs","Be","Mg","Ca","Sr","Ba","Sc","Y","Ti","Zr","Hf",
              "V","Nb","Ta","Cr","Mo","W","Mn","Tc","Re","Fe","Ru","Os","Co","Rh","Ir",
              "Ni","Pd","Pt","Cu","Ag","Au","Zn","Cd","Al","Ga","In","Tl","Sn","Pb","Bi",
              "Th"}
    nonmetals = {"H","C","N","O","F","Si","P","S","Cl","As","Se","Br","Te","I"}

    if n_el == 1:
        return "Pure elemental metal"
    if n_el == 2 and all(e in metals for e in elements):
        return "Binary intermetallic"
    if any(e in {"F","Cl","Br","I"} for e in elements):
        return "Halide"
    if "O" in elements:
        return "Oxide"
    if "C" in elements and n_atoms <= 4:
        return "Simple carbide"
    if n_el == 2 and any(e in nonmetals for e in elements):
        return "Simple binary compound (metal+nonmetal)"
    return "Other / complex compound"


def main():
    df = pd.read_csv(OUTLIER_CSV)
    df["id"] = df["id"].astype(str).str.strip()

    formulas, elements_list, n_atoms_list, sg_list, density_list, vpa_list, category_list = \
        [], [], [], [], [], [], []

    for mp_id in df["id"]:
        cif_path = find_cif_path(mp_id, CIF_DIR)
        if cif_path is None:
            formulas.append("CIF NOT FOUND")
            elements_list.append("")
            n_atoms_list.append("")
            sg_list.append("")
            density_list.append("")
            vpa_list.append("")
            category_list.append("")
            continue
        formula, elements, n_atoms, sg, density, vpa = get_formula_info(cif_path)
        formulas.append(formula)
        elements_list.append(elements)
        n_atoms_list.append(n_atoms)
        sg_list.append(sg)
        density_list.append(density)
        vpa_list.append(vpa)
        category_list.append(classify(elements, n_atoms if isinstance(n_atoms, int) else 0))

    df["formula"] = formulas
    df["elements"] = elements_list
    df["n_atoms"] = n_atoms_list
    df["spacegroup"] = sg_list
    df["density_g_cm3"] = density_list
    df["volume_per_atom"] = vpa_list
    df["rough_category"] = category_list

    print(df.to_string(index=False))

    print()
    print("Category counts among the 17 outliers:")
    print(df["rough_category"].value_counts().to_string())

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved full table to: {OUT_CSV}")


if __name__ == "__main__":
    main()