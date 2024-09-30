import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# Parameters for plots
FONTSIZE_TITLE = 18
FONTSIZE_AXES_LABEL = 14
FONTSIZE_LEGEND = 13
VAL_DPI = 300


def show_distribution(dt, names, name_set, show_plot=False):
    """
    Create a distribution plot given a column name in the Dataframe
    :param dt: Dataframe
    :param names: List of column name(s)
    :param name_set: String specifying if it's the training or testing set
    :param show_plot: Boolean to show the plot generated. Default: False
    """
    for column in names:
        # Create plot
        plt.figure(figsize=(10, 10))
        plt.hist(dt[column])

        # Labels
        title = f"Distribution of {column} in {name_set.replace('_', ' ')}"
        plt.title(title, fontsize=FONTSIZE_TITLE)
        plt.ylabel("Number of occurences", fontsize=FONTSIZE_AXES_LABEL)
        plt.xlabel("Concentration", fontsize=FONTSIZE_AXES_LABEL)
        ax = plt.gca()
        ax.tick_params(axis='both', which='both', length=5, labelsize=13)

        # Save plot
        path_file = f"plots/distribution/distribution_{name_set}_{column.lower()}.png"
        plt.savefig(path_file, dpi=VAL_DPI)
        if show_plot:
            plt.show()


if __name__ == "__main__":
    # Distribution in Training set
    train_path = Path("data/Table_S1_training_set.csv")
    train_df = pd.read_csv(train_path, sep=";")

    show_distribution(train_df, ["PGAM", "ENO", "PPDK", "Jpred"], "Training set")

    # Distribution in Testing set
    test_path = Path("data/Table_S2_test_set.csv")
    test_df = pd.read_csv(test_path, sep=";")

    set_names = ["Test set 1", "Test set 2", "Test set 3", "Test set 4", "Test set 5"]
    for i in range(len(set_names)):
        subset = test_df[test_df["Test set"] == set_names[i]]
        show_distribution(subset, ["PGAM", "ENO", "PPDK", "Jpred"], "Test_" + str(i) + " set")
