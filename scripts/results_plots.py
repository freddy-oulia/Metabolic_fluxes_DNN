import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import ptitprince as pt
from scipy import stats
from pathlib import Path


# Parameters for plots
FONTSIZE_TITLE = 18
FONTSIZE_AXES_LABEL = 14
FONTSIZE_LEGEND = 13
VAL_DPI = 300


def make_dataframe(data, list_metrics, cv_rho):
    names = ["Validation set", "Test set 1", "Test set 2", "Test set 3", "Test set 4", "Test set 5"]
    key_set_names = ["val_", "test_0_", "test_1_", "test_2_", "test_3_", "test_4_"]

    list_df = []
    # Iterate on each metric
    for metric_name in list_metrics:
        res_current_metric = []

        # Iterate on each set
        for current_set_name_id in range(len(key_set_names)):
            keyname_value = key_set_names[current_set_name_id] + metric_name

            # If cross-validation result
            if cv_rho == "cv":
                # Need to parkour every repetition and every fold to collect result
                for key, all_folds in data.items():
                    if key == "architecture":
                        pass
                    else:
                        for current_fold_name in all_folds:
                            res_current_metric.append([names[current_set_name_id],
                                                       all_folds[current_fold_name][keyname_value]])
            else:
                # else => repeated hold out result
                for key, current_repeat_dict in data.items():
                    if key == "architecture":
                        pass
                    else:
                        res_current_metric.append([names[current_set_name_id],
                                                   current_repeat_dict[keyname_value]])

        # Create dataframe with all result
        list_df.append(pd.DataFrame(res_current_metric, columns=["Set", metric_name.upper()]))
    return list_df


def make_boxplot(data_json, list_metrics, cv_rho):
    print("Generating boxplot...")
    sn.set(style="whitegrid", font_scale=2)

    # Load results into dataframe
    if cv_rho == "cv":
        dfs = make_dataframe(data_json, list_metrics, "cv")
        name = "Cross-validation"
        save_path = Path("plots/boxplot")/name.lower()
        save_path.mkdir(parents=True, exist_ok=True)
    else:
        dfs = make_dataframe(data_json, list_metrics, "rho")
        name = "Repeated hold out evaluation"
        save_path = Path("plots/boxplot")/name.lower().replace(" ", "_")
        save_path.mkdir(parents=True, exist_ok=True)

    for df in dfs:
        dx = df.columns[0]
        dy = df.columns[1]

        fig, ax = plt.subplots(figsize=(20, 15))
        ax = pt.RainCloud(
            x=dx,
            y=dy,
            data=df,
            palette="Set2",
            bw=.2,
            width_viol=.6,
            ax=ax,
            orient="v"
        )
        if cv_rho == "rho":
            if dy == "RMSE":
                y_scale = [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
                ax.set_yticks(y_scale)
            elif dy == "MAE":
                y_scale = [0.050, 0.055, 0.060, 0.065, 0.070, 0.075, 0.080, 0.085, 0.090, 0.095, 0.100]
                ax.set_yticks(y_scale)
            else:
                pass

        plt.title("Boxplot for " + dy + " - Results with " + name, fontsize=FONTSIZE_TITLE)
        ax.ticklabel_format(axis="y", style="scientific", useOffset=False)

        namefile = dy.lower() + ".png"
        path_save = save_path / namefile
        plt.savefig(path_save, bbox_inches='tight', dpi=VAL_DPI)


def get_all_result(dict_results, metric, cv_or_rho):
    """
    Retrieve result on specified metric on each test set
    :param dict_results: Dict with every result
    :param metric: String, name of the metric to retrieve result
    :param cv_or_rho: String, either 'cv' for repeated cross-validation or 'rho' for repeated hold-out evaluation
    :return List of list with the result of given metric of each test set
    """
    keys = ["test_" + str(i) + "_" + metric for i in range(5)]
    res = [[], [], [], [], []]

    # Loop over each result
    for repetition_loop in dict_results.keys():
        # One more loop for repeated cross-validation result
        if cv_or_rho == "cv":
            for fold_loop in dict_results[repetition_loop].keys():
                current_res = dict_results[repetition_loop][fold_loop]
                for i in range(len(keys)):
                    res[i].append(current_res[keys[i]])
        else:
            current_res = dict_results[repetition_loop]
            for i in range(len(keys)):
                res[i].append(current_res[keys[i]])
    return res


def compute_tau(list_res, name_correlation):
    """
    Compute tau (Kendall, Spearman or Pearson) on each pair of test set
    :param list_res: List with results on each test set for a given metric
    :param name_correlation: String, name of the correlation tau to use
    :return 2-d List with tau computed for each pair of test set
    """
    tau_res = []

    for line in range(5):
        res_line = []
        for column in range(5):
            if name_correlation == "Kendall":
                tau, _ = stats.kendalltau(list_res[line], list_res[column])
            elif name_correlation == "Spearman":
                tau, _ = stats.spearmanr(list_res[line], list_res[column])
            elif name_correlation == "Pearson":
                tau, _ = stats.pearsonr(list_res[line], list_res[column])

            res_line.append(tau)
        tau_res.append(res_line)
    return tau_res


def get_tau(dict_res, method):
    """
    Compute each correlation tau (Kendall, Spearman and Person) for each metric on test sets
    :param dict_res: Dict with every result
    :param method: String, either 'cv' for repeated cross-validation or 'rho' for repeated hold-out evaluation
    """
    print("Generating correlation metric heatmaps...")
    indexs = np.arange(1, 6)
    metrics_name = ["rmse", "r2", "mae"]
    taus_name = ["Kendall", "Spearman", "Pearson"]

    for correlation_name in taus_name:
        list_dataframe = []
        for metric in metrics_name:
            # Retrieve all results on each test set of a given metric
            metric_result = get_all_result(dict_res, metric, method)
            # Compute current correlation metric for current metric
            df_metric = pd.DataFrame(compute_tau(metric_result, correlation_name), index=indexs, columns=indexs)
            list_dataframe.append(df_metric)

        if correlation_name == "Kendall":
            part_title = "Kendall's Tau on "
            part_path = Path("plots/correlation/Kendall")
            part_path.mkdir(parents=True, exist_ok=True)

        elif correlation_name == "Spearman":
            part_title = "Spearman's rank correlation on "
            part_path = Path("plots/correlation/Spearman")
            part_path.mkdir(parents=True, exist_ok=True)

        else:
            part_title = "Pearson's correlation coefficient on "
            part_path = Path("plots/correlation/Pearson")
            part_path.mkdir(parents=True, exist_ok=True)

        # Create and save Heatmaps for each metric
        for i in range(len(list_dataframe)):
            plt.figure(figsize=(8, 8))
            heat_map = sn.heatmap(list_dataframe[i], annot=True, fmt=".3f", linewidth=0.5,
                                  square=True, annot_kws={"fontsize": 8})

            heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0)
            plt.xlabel("Test Set", fontsize=14)
            plt.ylabel("Test Set", fontsize=14)
            plt.title(part_title + metrics_name[i].upper(), fontsize=18)
            namefile = metrics_name[i] + "_" + method
            path_save = part_path / namefile
            plt.savefig(path_save, dpi=VAL_DPI)
