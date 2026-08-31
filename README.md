# CrysCo Benchmark for Poisson Ratio Prediction
## Overview
This repository contains the code, preprocessing pipeline, fixed dataset splits, target data, prediction outputs, and error analysis used to benchmark the \*\*CrysCo (Crystal Structure-based model)\*\* for predicting the \*\*Poisson ratio\*\* of crystalline materials.
The purpose of this benchmark is to evaluate the performance of CrysCo on a fixed Poisson-ratio dataset and provide a reproducible record of the preprocessing, model configuration, training outputs, and prediction analysis.

The repository was prepared as part of a research study involving machine-learning-based prediction of materials properties from crystal structures.

## Research Objective

The primary objective is to benchmark the CrysCo model for crystal-structure-based prediction of Poisson ratio.
The benchmark workflow is:
Crystal structures + target data
            |
             
   Data preprocessing
            |
          
  Feature and graph generation
            |
           
      CrysCo model
           |
          
        Model training
             |
             
    Test-set prediction
            |
           
    Error and outlier analysis
\---



## Dataset



The original Poisson-ratio dataset contains:



13,057 materials\*\* in the original selected dataset.

13,051 structures\*\* were successfully available for the CrysCo processing pipeline.
6 material IDs\*\* were removed during CrysCo processing. It failed in feature extraction process during data pre processing.



The usable dataset is divided into fixed training, validation, and test sets.



### Fixed Dataset Split



| Split                    | Number of materials |

| ------------------------ | ------------------: |

| Training                 |               9,404 |

| Validation               |               1,305 |

| Test                     |               2,342 |

| \*\*Total usable dataset\*\* |          \*\*13,051\*\* |



The fixed split files are provided in:



```text

split/poisson_ratio/




Files:



```text

crysco_train_ids.csv

crysco_val_ids.csv

crysco_test_ids.csv





Using fixed split files ensures that subsequent experiments can use the same train/validation/test assignment.



\---



## Target Data



The cleaned Poisson-ratio target data is provided in:



```text

targets/poisson_ratio_cleaned/





The directory contains:



```text

poisson_composition_final.csv

poisson_structural.csv





These files contain the property information used in the benchmark and preprocessing workflow.



\---



## CrysCo Model



The CrysCo implementation used in this benchmark is contained in:



```text

crysco_benchmark/models/




The implementation includes:



```text

CrysCo.py

EGAT.py

MLP.py

SE.py

transformer.py





The model combines graph-based representations of crystal structures with additional feature-processing components.



The supporting data and utility implementations are located in:



```text

crysco_benchmark/data/

crysco_benchmark/utils/





\---



## Preprocessing Pipeline



The preprocessing scripts are located in:



```text

scripts/preprocessing/




Important files include:



```text

data_preparation.py

extracted_features.py

extracted_features_diagnostic.py

graph_dihedral.py





The preprocessing pipeline converts crystal structure and property information into a PyTorch Geometric representation suitable for CrysCo training.



The pipeline includes:



\* crystal structure loading,

\* atomic feature generation,

\* graph construction,

\* distance-based edge generation,

\* Gaussian edge features,

\* angle features,

\* dihedral-angle features,

\* global atomic features,

\* target-property assignment,

\* PyTorch Geometric data construction.



The resulting processed dataset is stored locally as a PyTorch `.pt` file.



\---



## Training Configuration



The benchmark training configuration used for the 100-epoch experiment includes:



| Parameter             |             Value |

| --------------------- | ----------------: |

| Model                 |            CrysCo |

| Epochs                |               100 |

| Batch size            |                80 |

| Learning rate         |             0.006 |

| Optimizer             |             AdamW |

| Scheduler             | ReduceLROnPlateau |

| Activation            |              SiLU |

| Dropout               |               0.0 |

| Output dimensions     |                64 |

| Transformer dimension |               128 |

| Transformer layers    |                 3 |

| Attention heads       |                 4 |

| EGAT layers           |                 5 |

| Pooling               |   global\_add\_pool |

| Pooling order         |             early |



The benchmark was trained using GPU acceleration when available. the script was Train\_crysco\_poisson\_100ep.py



\---



## Benchmark Results



The 100-epoch CrysCo benchmark was evaluated on the fixed test set containing \*\*2,342 materials\*\*.



The principal test-set metrics obtained were:



| Metric        |       Value |

| ------------- | ----------: |

| MAE           | \*\*0.231437\*\* |

| RMSE          | \*\*0.673353\*\* |

| Target median | \*\*0.30000\*\* |

| MAD           | \*\*0.06385\*\* |

| MAE / MAD     | \*\*3.621145\*\* |



The prediction outputs are provided in:



```text

results/





The main test output is:



```text

crysco_poisson_100ep_test_outputs.csv





Training and validation outputs are also provided.



\---



## Prediction and Error Analysis



The repository contains additional analysis of the CrysCo predictions.



### Available analysis files



```text

results/

├── best_worst_with_formulas.csv

├── crysco_poisson_100ep_test_outputs.csv

├── crysco_poisson_100ep_train_outputs.csv

├── crysco_poisson\_100ep\_val\_outputs.csv

├── outlier_17_full_analysis.csv

└── worst_offenders_ranked.csv





These files were used to investigate:



\* largest prediction errors,

\* best and worst predictions,

\* predicted versus actual Poisson ratios,

\* material formulas associated with extreme errors,

\* problematic/outlier structures,

\* categories of structures associated with poor predictions.



Additional structural analysis is provided in:



```text

structure\_analysis/




\---



## Analysis Scripts



The main analysis scripts are:



```text

scripts/

├── train.py

├── find_worst_offenders.py

├── get_formulas_best_worst.py

├── analyze_outlier_categories.py

└── preprocessing/





Additional benchmark-specific scripts are located in the repository root:



```text

Train_crysco_poisson_100ep.py

compute_mae_mad_ratio.py





\---



## Repository Structure



```text

crysco_benchmark/

│

├── README.md

├── .gitignore

├── requirements.txt

│

├── crysco_benchmark/

│   ├── data/

│   ├── models/

│   └── utils/

│

├── scripts/

│   ├── train.py

│   ├── find_worst_offenders.py

│   ├── get_formulas_best_worst.py

│   ├── analyze_outlier_categories.py

│   └── preprocessing/

│

├── targets/

│   └── poisson_ratio_cleaned/

│

├── split/

│   └── poisson_ratio/

│

├── results/

│

└── structure_analysis/





\---



## Files Not Included in the Repository



Large generated files and raw crystal structures are intentionally excluded from GitHub.



### Crystal structure CIF files



The original CIF structures are not included.



Expected local directory:



```text

structures/

└── poisson_ratio_final/




These structures were used during preprocessing to generate the graph representations required by CrysCo.



### Processed PyTorch dataset



The generated processed dataset:



```text

processed/poisson_ratio.pt





is not included because of its large file size (approximately \*\*397 MB\*\*).



It can be generated from the preprocessing pipeline using the provided preprocessing scripts.



### Trained model checkpoints



The trained model checkpoints are also not included:



```text

models/crysco_poisson_100ep_best.pth

models/crysco_smoke_test.pth





The main 100-epoch checkpoint is approximately \*\*149 MB\*\*.



The checkpoints were retained locally and are excluded from the GitHub repository to keep the repository lightweight.



\---



## Reproducibility



To reproduce the benchmark, the following components are required:



1. The crystal structure CIF files.

2. The target CSV files provided in this repository.

3. The fixed train/validation/test split files.

4. The preprocessing scripts.

5. The CrysCo model implementation.

6. The required Python dependencies.

7. The generated processed PyTorch dataset.



The recommended workflow is:



```text

1. Obtain the required CIF structures
2. Place structures in structures/poisson_ratio_final/
3. Use target files from targets/poisson_ratio_cleaned/
4. Run scripts/preprocessing/data_preparation.py
5. Generate processed/poisson_ratio.pt
6. Train CrysCo
7. Evaluate using the fixed split
8. Compare predictions with results/





\---



## Software Environment



The benchmark was developed and tested using Python and the following major scientific and machine-learning packages:



```text

Python

PyTorch

PyTorch Geometric

ASE

pymatgen

matminer

NumPy

pandas

SciPy

scikit-learn

Matplotlib

```



The main dependencies are listed in:



```text

requirements.txt

```



\---



## Important Note About Data Splits



The benchmark should be interpreted using the fixed split files provided in:



```text

split/poisson_ratio/

```



These files were prepared to maintain a consistent train/validation/test assignment for model comparison.



The CrysCo usable dataset contains 13,051 structures, consisting of:



```text

Training:   9,404

Validation: 1,305

Test:       2,342

```



\---



## Purpose of the Repository



This repository is intended to provide a transparent record of the CrysCo Poisson-ratio benchmark, including:



\* model implementation,

\* preprocessing methodology,

\* target data,

\* fixed dataset splits,

\* training configuration,

\* prediction outputs,

\* evaluation metrics,

\* outlier analysis,

\* structural analysis.



Large raw and generated files are intentionally kept outside the repository.



\---



## Status



\*\*Benchmark status:\*\* Completed for the 100-epoch CrysCo Poisson-ratio experiment.



Further analysis and comparison with other graph neural network models may be added as the research progresses.



