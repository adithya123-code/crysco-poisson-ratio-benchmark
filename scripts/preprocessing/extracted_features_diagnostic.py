import os
import csv
import numpy as np
import torch

from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN


def human_features(directory_path, csv_file):

    # ---------------------------------------------------------
    # Use absolute paths
    # ---------------------------------------------------------
    directory_path = os.path.abspath(directory_path)
    csv_file = os.path.abspath(csv_file)

    print("==========================================")
    print("Human Feature Diagnostic")
    print("==========================================")
    print("CIF directory :", directory_path)
    print("CSV file      :", csv_file)
    print()

    # ---------------------------------------------------------
    # Check paths
    # ---------------------------------------------------------
    if not os.path.isdir(directory_path):
        raise FileNotFoundError(
            f"CIF directory does not exist: {directory_path}"
        )

    if not os.path.isfile(csv_file):
        raise FileNotFoundError(
            f"CSV file does not exist: {csv_file}"
        )

    # ---------------------------------------------------------
    # Read CSV
    # ---------------------------------------------------------
    with open(csv_file, newline="") as f:
        reader = csv.reader(f)
        target_data = list(reader)

    target_data = target_data[1:]

    print("CSV structures:", len(target_data))

    # ---------------------------------------------------------
    # Check CIF availability
    # ---------------------------------------------------------
    missing_cifs = []

    for row in target_data:
        structure_id = row[0]
        cif_file = os.path.join(
            directory_path,
            structure_id + ".cif"
        )

        if not os.path.isfile(cif_file):
            missing_cifs.append(structure_id)

    print("Missing CIFs   :", len(missing_cifs))

    if missing_cifs:
        print("\nFirst missing CIFs:")
        for structure_id in missing_cifs[:20]:
            print(" ", structure_id)

    print()

    # ---------------------------------------------------------
    # Create featurizers
    # ---------------------------------------------------------
    density_featurizer = DensityFeatures()
    global_symmetry_featurizer = GlobalSymmetryFeatures()
    voronoi_nn = VoronoiNN()

    print("Featurizers initialized.")
    print()

    # ---------------------------------------------------------
    # Process structures
    # ---------------------------------------------------------
    data_list = []
    error_indices = []

    for index, row in enumerate(target_data):

        structure_id = row[0]

        cif_file = os.path.join(
            directory_path,
            structure_id + ".cif"
        )

        print(
            f"[{index + 1}/{len(target_data)}] "
            f"Processing {structure_id}",
            flush=True
        )

        if not os.path.isfile(cif_file):
            print("  SKIPPED: CIF missing", flush=True)
            error_indices.append(index)
            continue

        try:

            # -------------------------------------------------
            # Read structure
            # -------------------------------------------------
            structure = Structure.from_file(cif_file)

            # -------------------------------------------------
            # Density features
            # -------------------------------------------------
            density_features = density_featurizer.featurize(
                structure
            )

            # -------------------------------------------------
            # Global symmetry features
            # -------------------------------------------------
            global_symmetry_features = (
                global_symmetry_featurizer.featurize(structure)
            )

            # -------------------------------------------------
            # Basic structural information
            # -------------------------------------------------
            volume = structure.volume

            primitive_structure = (
                SpacegroupAnalyzer(structure).find_primitive()
            )

            num_atoms_primitive_cell = len(
                primitive_structure
            )

            num_atoms = len(structure)

            volume_per_atom = volume / num_atoms

            # -------------------------------------------------
            # Atomic radii
            # -------------------------------------------------
            atom_radii = [
                element.atomic_radius
                for element in structure.species
            ]

            total_atom_volume = sum(
                (4.0 / 3.0) * np.pi * (radius ** 3)
                for radius in atom_radii
            )

            packing_fraction = (
                total_atom_volume / volume
            )

            # -------------------------------------------------
            # Lattice parameters
            # -------------------------------------------------
            lattice_parameters = (
                structure.lattice.parameters
            )

            # -------------------------------------------------
            # Voronoi coordination
            # -------------------------------------------------
            voronoi_polyhedra = (
                voronoi_nn.get_all_voronoi_polyhedra(
                    structure
                )
            )

            voronoi_coord_numbers = [
                len(voronoi)
                for voronoi in voronoi_polyhedra
            ]

            mean_voronoi_coord_number = np.mean(
                voronoi_coord_numbers
            )

            std_dev_voronoi_coord_number = np.std(
                voronoi_coord_numbers
            )

            # -------------------------------------------------
            # Bond calculations
            # -------------------------------------------------
            bond_angles = []
            bond_lengths = []
            neighbor_distances = []

            for i, site in enumerate(structure):

                neighbors = voronoi_nn.get_nn_info(
                    structure,
                    i
                )

                for neighbor_info in neighbors:

                    central_coord = site.coords
                    neighbor_coord = (
                        neighbor_info["site"].coords
                    )

                    bond_vector = (
                        central_coord - neighbor_coord
                    )

                    bond_length = np.linalg.norm(
                        bond_vector
                    )

                    bond_lengths.append(bond_length)
                    neighbor_distances.append(bond_length)

                    for second_neighbor_info in neighbors:

                        if second_neighbor_info == neighbor_info:
                            continue

                        second_neighbor_coord = (
                            second_neighbor_info["site"].coords
                        )

                        bond_vector_1 = (
                            central_coord - neighbor_coord
                        )

                        bond_vector_2 = (
                            central_coord
                            - second_neighbor_coord
                        )

                        norm1 = np.linalg.norm(
                            bond_vector_1
                        )

                        norm2 = np.linalg.norm(
                            bond_vector_2
                        )

                        if norm1 != 0 and norm2 != 0:

                            cosine = (
                                np.dot(
                                    bond_vector_1,
                                    bond_vector_2
                                )
                                / (norm1 * norm2)
                            )

                            cosine = np.clip(
                                cosine,
                                -1.0,
                                1.0
                            )

                            angle = np.arccos(cosine)

                            bond_angles.append(
                                np.degrees(angle)
                            )

            # -------------------------------------------------
            # Statistical descriptors
            # -------------------------------------------------
            mean_avg_bond_angle = np.nanmean(
                bond_angles
            )

            std_dev_avg_bond_angle = np.nanstd(
                bond_angles
            )

            mean_avg_bond_length = np.mean(
                bond_lengths
            )

            std_dev_avg_bond_length = np.std(
                bond_lengths
            )

            mean_neighbor_distance = np.mean(
                neighbor_distances
            )

            std_dev_neighbor_distance = np.std(
                neighbor_distances
            )

            min_neighbor_distance = np.min(
                neighbor_distances
            )

            max_neighbor_distance = np.max(
                neighbor_distances
            )

            # -------------------------------------------------
            # Lattice parameters
            # -------------------------------------------------
            a, b, c, alpha, beta, gamma = (
                lattice_parameters
            )

            # -------------------------------------------------
            # Global symmetry features
            # -------------------------------------------------
            gs_features = [
                global_symmetry_features[0],
                global_symmetry_features[2],
                global_symmetry_features[4]
            ]

            # -------------------------------------------------
            # Final human-designed feature vector
            # -------------------------------------------------
            structure_vector = [
                volume,
                num_atoms_primitive_cell,
                num_atoms,
                volume_per_atom,
                packing_fraction,
                mean_voronoi_coord_number,
                std_dev_voronoi_coord_number,
                mean_avg_bond_angle,
                std_dev_avg_bond_angle,
                mean_avg_bond_length,
                std_dev_avg_bond_length,
                mean_neighbor_distance,
                std_dev_neighbor_distance,
                min_neighbor_distance,
                max_neighbor_distance,
                a,
                b,
                c,
                alpha,
                beta,
                gamma,
                *gs_features
            ]

            data_list.append(structure_vector)

            print(
                f"  SUCCESS: {len(structure_vector)} features",
                flush=True
            )

        except Exception as e:

            error_indices.append(index)

            print(
                f"  ERROR: {type(e).__name__}: {e}",
                flush=True
            )

        # -----------------------------------------------------
        # Progress summary
        # -----------------------------------------------------
        if (index + 1) % 10 == 0:

            print(
                f"\nProgress: {index + 1}/"
                f"{len(target_data)}"
            )

            print(
                f"Successful: {len(data_list)}"
            )

            print(
                f"Errors: {len(error_indices)}"
            )

            print()

    # ---------------------------------------------------------
    # Final diagnostic summary
    # ---------------------------------------------------------
    print()
    print("==========================================")
    print("DIAGNOSTIC SUMMARY")
    print("==========================================")
    print("CSV rows          :", len(target_data))
    print("Features generated:", len(data_list))
    print("Errors            :", len(error_indices))
    print("Missing CIFs      :", len(missing_cifs))
    print("==========================================")

    if error_indices:
        print("\nFirst error indices:")

        for i in error_indices[:20]:
            print(
                i,
                target_data[i][0]
            )

    return data_list


if __name__ == "__main__":

    cif_directory = (
        "/home/rajasekarakumar.v/workspace/adithya/"
        "Crysco/structures/poisson_ratio"
    )

    structural_csv = (
        "/home/rajasekarakumar.v/workspace/adithya/"
        "Crysco/targets/poisson_structural.csv"
    )

    features = human_features(
        cif_directory,
        structural_csv
    )

    print()
    print(
        "Final human feature count:",
        len(features)
    )
