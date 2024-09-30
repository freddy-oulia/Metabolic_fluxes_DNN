import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from pathlib import Path


# Parameters for plots
FONTSIZE_TITLE = 18
FONTSIZE_AXES_LABEL = 14
FONTSIZE_LEGEND = 13
VAL_DPI = 300


def evaluate_pls(pls_model, x_set, y_true, bool_norm, norm_value):
    """
    Function to evaluate pls model on a given set.
    :param pls_model: PLS fit model
    :param x_set: Features for pls model
    :param y_true: Expected values
    :param bool_norm: Boolean if normalization was used on outputs
    :param norm_value: Values to reverse the normalization on outputs
    :returns: Results on RMSE, MAE, R2 and MSE
    """
    # Prediction
    y_pred = pls_model.predict(x_set)

    # De-normalize
    if bool_norm:
        y_pred = y_pred * norm_value
        y_true = y_true * norm_value

    # Compute metrics
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=True)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    return rmse, mae, r2, mse


def make_predict_vs_expected_plot(pls_model, x_test, y_test, bool_norm, norm_outputs, showfig, save_fig, save_path):
    """
    Make plot to compare predicted values with expected values with given model
    :param pls_model: PLS trained model
    :param x_test: Test set features
    :param y_test: Test set outputs
    :param bool_norm: Boolean, True if the outputs are normalized
    :param norm_outputs: Values used to normalize outputs
    :param showfig: Boolean to show the figure
    :param save_fig: Boolean to save the figure
    :param save_path: String path to save the figure
    """
    # Store predicted and expected values
    expected_values = []
    predicted_values = []

    # Concatenate and de-normalize
    for i in range(len(x_test)):
        if bool_norm:
            expected_values.extend((y_test[i] * norm_outputs).tolist())
            predicted_values.extend((pls_model.predict(x_test[i]) * norm_outputs).tolist())
        else:
            expected_values.extend(y_test[i].tolist())
            predicted_values.extend(pls_model.predict(x_test[i]).tolist())

    # Need min and max for the straight line
    min_value = min(min(np.array(expected_values)), min(np.array(predicted_values)), 999999)
    max_value = max(max(np.array(expected_values)), max(np.array(predicted_values)), -999999)
    limit = [math.floor(min_value) - 1, math.ceil(max_value) + 1]

    # Create Figure
    plt.figure(figsize=(17, 12))

    # Draw straight line
    plt.plot(limit, limit, color="darkred", linestyle="solid")

    # Add points
    plt.scatter(expected_values, predicted_values, c="darkcyan", alpha=0.5, s=100, marker='o', label="Test set")

    # Labels
    plt.ylabel("Predicted Jpred (nmol/min)", fontsize=FONTSIZE_AXES_LABEL)
    plt.xlabel("Observed Jpred (nmol/min)", fontsize=FONTSIZE_AXES_LABEL)
    plt.title("Predicted vs expected values", fontsize=FONTSIZE_TITLE)

    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', which='both', length=5, labelsize=13)
    ax.yaxis.set_ticks(range(0, math.ceil(max_value), 10), minor=True)
    ax.xaxis.set_ticks(range(0, math.ceil(max_value), 10), minor=True)
    ax.tick_params(which='both', width=2)
    ax.tick_params(which='major', length=10)
    ax.tick_params(which="minor", length=4)
    plt.grid()

    # Save figure
    if save_fig:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path/"pls_predicted_and_observed.png", dpi=VAL_DPI)

    # Show figure
    if showfig:
        plt.show()
    return


def run_model(path_train_set, path_test_set, norm_output=False, predicted_expected_plot=True):
    """
    Training and evaluating PLS model
    :param path_train_set: String path to training set
    :param path_test_set: String path to test set
    :param norm_output: Boolean to normalize output
    :param predicted_expected_plot: Boolean to make predicted vs expected plot
    """
    # Load and normalize train set
    train_dataset = pd.read_csv(path_train_set, sep=";")
    np_df = train_dataset.to_numpy()
    max_features = np.max(np_df[:, :3])
    max_outputs = np.max(np_df[:, 3])

    # Normalization
    x_train = np_df[:, :3] / max_features
    if norm_output:
        y_train = np_df[:, 3] / max_outputs
    else:
        y_train = np_df[:, 3]

    # Concatenate X and Y of training set to make validation set
    training_set = np.concatenate((x_train, y_train[:, np.newaxis]), axis=1)
    x_train, x_validation = train_test_split(training_set, test_size=0.2, random_state=42)
    y_train = x_train[:, 3]
    x_train = x_train[:, :3]
    y_validation = x_validation[:, 3]
    x_validation = x_validation[:, :3]

    # Load and normalize test sets
    test_dataset = pd.read_csv(path_test_set, sep=";")
    np_df_test = test_dataset.to_numpy()
    xs_test = []
    ys_test = []

    set_names = ["Test set 1", "Test set 2", "Test set 3", "Test set 4", "Test set 5"]
    for i in range(len(set_names)):
        subset = np_df_test[np.where(np_df_test[:, 0] == set_names[i])]
        subset = subset[:, 1:].astype(np.float32)

        # Normalize with training set information
        xs_test.append(subset[:, :3] / max_features)
        if norm_output:
            ys_test.append(subset[:, 3] / max_outputs)
        else:
            ys_test.append(subset[:, 3])

    # PLS model
    pls1 = PLSRegression()
    # Training
    pls1.fit(x_train, y_train)

    # Performance on validation set
    val_rmse, val_mae, val_r2, _ = evaluate_pls(pls_model=pls1, x_set=x_validation, y_true=y_validation,
                                                bool_norm=norm_output, norm_value=max_outputs)
    print("Performance on validation set: ")
    print(f"RMSE: {val_rmse}; MAE: {val_mae}; R2: {val_r2}")

    # Performance on test sets
    avg_mse = 0.0
    avg_r2 = 0.0
    avg_mae = 0.0

    for i in range(len(set_names)):
        test_rmse, test_mae, test_r2, test_mse = evaluate_pls(pls_model=pls1, x_set=xs_test[i], y_true=ys_test[i],
                                                              bool_norm=norm_output, norm_value=max_outputs)

        avg_mse += test_mse
        avg_mae += test_mae
        avg_r2 += test_r2

        print(f"\nPerformances on testing set {i}: ")
        print(f"RMSE : {test_rmse}; MAE: {test_mae}; R2: {test_r2}")

    print(f"\nAverage performances on {len(set_names)} test sets:")
    print(f"Mean RMSE: {math.sqrt(avg_mse / len(set_names))}; Mean MAE: {avg_mae / len(set_names)}; Mean R2: {avg_r2 / len(set_names)}")

    # Make plot that compares expected values and predicted values
    if predicted_expected_plot:
        make_predict_vs_expected_plot(pls_model=pls1, x_test=xs_test, y_test=ys_test, bool_norm=norm_output,
                                      norm_outputs=max_outputs, showfig=True, save_fig=False,
                                      save_path="plots/pls_model")
    return


if __name__ == "__main__":
    run_model(path_train_set="data/Table_S1_training_set.csv", path_test_set="data/Table_S2_test_set.csv",
              norm_output=True, predicted_expected_plot=True)
