"""
Compute MAE, MSE, RMSE, R^2, and MAE/MAD ratio from a CrysCo *_test_outputs.csv
(as written by crysco.utils.utils_train.write_results).

MAD = mean absolute deviation of the TEST targets from their own mean.
      i.e. the error of a trivial "always predict the mean" baseline.
MAE/MAD ratio < 1  -> model beats the mean-baseline (lower is better, 0 = perfect)
MAE/MAD ratio ~= 1 -> model is basically no better than guessing the mean
MAE/MAD ratio > 1  -> model is worse than guessing the mean
(This is the same convention CGCNN's own repo/paper reports it in.)
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV_PATH = r"D:\MyProjects\crysco_benchmark\crysco_poisson_100ep_test_outputs.csv"  # <-- edit if needed


def main(csv_path):
    df = pd.read_csv(csv_path)

    # write_results always names the 3 columns: ids, target, prediction
    df.columns = [c.strip().lower() for c in df.columns]
    target = df["target"].astype(float).values
    pred = df["prediction"].astype(float).values

    n = len(df)
    mae = mean_absolute_error(target, pred)
    mse = mean_squared_error(target, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(target, pred)

    mad = np.mean(np.abs(target - np.mean(target)))
    mae_mad_ratio = mae / mad

    print(f"N test samples : {n}")
    print(f"MAE            : {mae:.6f}")
    print(f"MSE            : {mse:.6f}")
    print(f"RMSE           : {rmse:.6f}")
    print(f"R^2            : {r2:.6f}")
    print(f"MAD (baseline) : {mad:.6f}")
    print(f"MAE/MAD ratio  : {mae_mad_ratio:.6f}")

    return mae, mse, rmse, r2, mad, mae_mad_ratio


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else CSV_PATH
    main(path)