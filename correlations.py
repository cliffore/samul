#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, pointbiserialr
import json
import datetime
import traceback
from pathlib import Path
from utils import *


# ----------------------------
# Helpers
# ----------------------------

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")

def cohens_d_two_groups(pos: np.ndarray, neg: np.ndarray) -> float:
    n1, n0 = len(pos), len(neg)
    if n1 < 2 or n0 < 2:
        return float("nan")
    m1, m0 = pos.mean(), neg.mean()
    s1, s0 = pos.std(ddof=1), neg.std(ddof=1)
    sp = np.sqrt(((n1-1)*s1**2 + (n0-1)*s0**2) / (n1+n0-2)) if (n1+n0-2) > 0 else 0.0
    return float((m1 - m0) / sp) if sp > 0 else float("nan")

# ----------------------------
# Main
# ----------------------------

def calculate_correlations(thisProcess, log_file, similarities_folder, this_exp_folder):

    try:
        # ---- Read config ----
        with open("config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        experiment_folder = this_exp_folder

        # Output header (covers both binary & continuous cases)
        out_rows = []
        header = [
            "Activation",
            "Layer",
            "Sparsity",
            "Language",
            "Dictionary",
            "MeanMethod",
            "OriginalDatasetSize",
            "NonzeroGround_truth",
            "ZeroGround_truth",
            "BinaryDetected",
            "BalancedDatasetSize",
            # Binary metrics
            "PointBiserial_r",
            "PointBiserial_p",
            "ROC_AUC",
            "Cohens_d",
            # Continuous metrics
            "Pearson_r",
            "Pearson_p",
            "Spearman_r",
            "Spearman_p",
        ]

        data_folder = similarities_folder
        for sim_file in Path(data_folder).iterdir():
            if '.DS_Store' not in sim_file.name:

                csv_path = data_folder + '/' + sim_file.name

                language = ''

                if '--en--' in sim_file.name:
                    language = 'en'
                elif '--fr--' in sim_file.name:
                    language = 'fr'
                elif '--zh-CN--' in sim_file.name:
                    language = 'zh'
                    

                activation_method = ''

                if '--max-per-concept--' in sim_file.name:
                    activation_method = 'max-per-concept'
                elif '--mean-per-concept--' in sim_file.name:
                    activation_method = 'mean-per-concept'
                elif '--sum-per-concept--' in sim_file.name:
                    activation_method = 'sum-per-concept'
                elif '--top-1-per-token--' in sim_file.name:
                    activation_method = 'top-1-per-token'
                elif 'resid_post_mlp' in sim_file.name:
                    activation_method = 'resid_post_mlp'


                layer = sim_file.name.split('layer_')[1].split('__width_16')[0]
                sparsity = 'unknown'
                if 'average_l0_' in sim_file.name:
                    sparsity = sim_file.name.split('average_l0_')[1].replace('.csv', '')
                dictionary = 'unknown'
                if '__width_' in sim_file.name:
                    dictionary = sim_file.name.split('__width_')[1].split('__average_l0')[0]
                elif '_v5_' in sim_file.name:
                    dictionary = sim_file.name.split('_v5_')[1].split('--')[0]


                mean_method = 'none'

                if '--full_mean--' in sim_file.name:
                    mean_method = 'full_mean'
                elif '--geometric_mean--' in sim_file.name:
                    mean_method = 'geometric_mean'
                elif '--harmonic_mean--' in sim_file.name:
                    mean_method = 'harmonic_mean'
                elif '--intersection_mean--' in sim_file.name:
                    mean_method = 'intersection_meann'



                # Load
                df = pd.read_csv(csv_path, encoding="utf-8")

                # Basic columns check
                if not {"similarity", "ground_truth"}.issubset(df.columns):
                    update_log(log_file, thisProcess + f": Skipping {sim_file.name} (missing required columns).")
                    continue

                # Stats for reporting
                total_n = len(df)
                nonzero = df[df["ground_truth"] > 0]
                zeros   = df[df["ground_truth"] == 0]
                # Balanced size for info only (we do correlations on FULL set)
                zeros_sampled = zeros.sample(n=min(len(zeros), len(nonzero)), random_state=42) if len(nonzero) > 0 else zeros.head(0)
                balanced_df = pd.concat([nonzero, zeros_sampled], ignore_index=True)

                # Extract arrays and clean
                distances = pd.to_numeric(df["similarity"], errors="coerce").to_numpy()
                gt = pd.to_numeric(df["ground_truth"], errors="coerce").to_numpy()

                mask = np.isfinite(distances) & np.isfinite(gt)
                distances = distances[mask]
                gt = gt[mask]

                # Initialize outputs
                binary_detected = False
                pb_r = pb_p = auc = d_val = np.nan
                pear_r = pear_p = spear_r = spear_p = np.nan

                # Decide binary vs continuous
                if is_binary_array(gt, tol=0.0):
                    binary_detected = True
                    # Use similarity so higher = “more similar”; correlation expected to be positive if GT=1 means similar
                    similarity = distances

                    if similarity.size == 0 or gt.size == 0:
                        pb_r = pb_p = auc = d_val = np.nan
                    else:
                        # Check both classes present
                        ones = int((gt == 1).sum())
                        zeros_cnt = int((gt == 0).sum())
                        if ones == 0 or zeros_cnt == 0:
                            pb_r = pb_p = auc = d_val = np.nan
                        else:
                            # Point-biserial (equiv to Pearson(similarity, binary_label))
                            try:
                                pb_r, pb_p = pointbiserialr(gt.astype(int), similarity.astype(float))
                            except Exception:
                                pb_r = pb_p = np.nan

                            # ROC-AUC
                            auc = safe_auc(gt.astype(int), similarity.astype(float))

                            # Cohen's d
                            pos = similarity[gt == 1]
                            neg = similarity[gt == 0]
                            d_val = cohens_d_two_groups(pos, neg)
                else:
                    # Continuous: Pearson & Spearman on FULL set
                    # Guard against zero variance (Pearson/Spearman would return nan/raise)
                    if distances.size >= 2 and gt.size >= 2 and np.std(distances) > 0 and np.std(gt) > 0:
                        try:
                            pear_r, pear_p = pearsonr(distances, gt)
                        except Exception:
                            pear_r = pear_p = np.nan
                        try:
                            spear_r, spear_p = spearmanr(distances, gt)
                        except Exception:
                            spear_r = spear_p = np.nan

                out_rows.append([
                    activation_method,
                    layer,
                    sparsity,
                    language,
                    dictionary,
                    mean_method,
                    total_n,
                    int((df["ground_truth"] > 0).sum()),
                    int((df["ground_truth"] == 0).sum()),
                    int(binary_detected),
                    len(balanced_df),
                    # Binary metrics (may be NaN for continuous)
                    pb_r, pb_p, auc, d_val,
                    # Continuous metrics (may be NaN for binary)
                    pear_r, pear_p, spear_r, spear_p
                ])

        # Write results
        out_csv = os.path.join(experiment_folder, "correlations.csv")
        out_df = pd.DataFrame(out_rows, columns=header)
        # Use a stable float format
        out_df.to_csv(out_csv, index=False, float_format="%.8f", encoding="utf-8")
        print(f"Wrote: {out_csv}")

        update_log(log_file, thisProcess + "Wrote correlations to :" + out_csv)
        update_log(log_file, thisProcess + ": end")
        print("Done.")

    except Exception as e:
        update_log(log_file, thisProcess + ": Exception = " + str(e))
        update_log(log_file, thisProcess + ": Exception = " + traceback.format_exc())
        print("ERROR:", e)