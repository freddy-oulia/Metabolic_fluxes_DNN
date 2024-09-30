import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
import ptitprince as pt
import seaborn as sn
from pathlib import Path


NAME_METRICS = ["mse", "rmse", "r2", "mae"]

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


def boxplots_comparison(data_json_cv, data_json_rho, list_metrics):
    """
    Make boxplot comparison between result from cross validation and repeated hold out evaluation
    :param data_json_cv: Json file with cross-validation results
    :param data_json_rho: Json file with repeated hold out evaluation result
    :param list_metrics: List of metrics name to make boxplot
    """
    sn.set(style="whitegrid", font_scale=2)
    # Load results into dataframe

    dfs_cv = make_dataframe(data_json_cv, list_metrics, "cv")
    dfs_rho = make_dataframe(data_json_rho, list_metrics, "rho")

    save_path = Path("plots/boxplot/comparison")
    save_path.mkdir(parents=True, exist_ok=True)

    for df_index in range(len(dfs_cv)):
        dfs_cv[df_index]["method"] = "repeated cross-validation"
        dfs_rho[df_index]["method"] = "repeated hold-out evaluation"
        df_comparison = pd.concat([dfs_cv[df_index], dfs_rho[df_index]])

        dx = df_comparison.columns[0]
        dy = df_comparison.columns[1]
        dhue = df_comparison.columns[2]

        f, ax = plt.subplots(figsize=(20, 20))

        ax = pt.RainCloud(x=dx, y=dy, hue=dhue, data=df_comparison, palette="Set2", bw=.2, width_viol=.7, ax=ax,
                          orient="h", alpha=.65, dodge=True)

        title = list_metrics[df_index].upper() + " - Comparison results between repeated cross-validation and repeated hold-out evaluation procedures"
        path_file = save_path / f"{list_metrics[df_index]}.png"

        plt.title(title)
        ax.ticklabel_format(axis="x", style="scientific", useOffset=False)
        plt.savefig(path_file, bbox_inches='tight', dpi=VAL_DPI)


if __name__ == "__main__":
    # Get cross-validation result file
    path = Path("results/cv/performances_cv.json")
    if path.is_file():
        with open(path, "r") as fp:
            file_cv = json.load(fp)
    else:
        sys.exit("Cross validation result file not found. Cannot make the boxplot comparison")

    # Get repeated hold out evaluation result file
    path = Path("results/rho/performances_rho.json")
    if path.is_file():
        with open(path, "r") as fp:
            file_rho = json.load(fp)
    else:
        sys.exit("Repeated hold-out evaluation result file not found. Cannot make the boxplot comparison")

    # Make the boxplot comparison with given metrics
    list_metrics = ["rmse", "mae", "r2"]
    boxplots_comparison(file_cv, file_rho, list_metrics)
