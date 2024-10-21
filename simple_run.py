import time
import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from numpy.random import seed
from sklearn.model_selection import train_test_split
from metrics import R2
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


# Parameters for plots
FONTSIZE_TITLE = 18
FONTSIZE_AXES_LABEL = 14
FONTSIZE_LEGEND = 13
VAL_DPI = 300


def print_performance(x_set, y_set, model, max_outputs, name_set):
    """
    Print model performance on a given set.
    :param x_set: features set
    :param y_set: outputs set
    :param model: trained model
    :param max_outputs: values to de-normalize outputs
    :param name_set: String, name of the given set
    """
    # Get predictions
    if name_set == "Test":
        start_time = time.time()
        predictions = model.predict(x=x_set, verbose=0)
        print(f"Inference time on test set {time.time() - start_time} seconds")
    else:
        predictions = model.predict(x=x_set, verbose=0)

    # De-normalize outputs
    predicted_values = predictions * max_outputs
    expected_values = y_set * max_outputs

    # Compute metrics
    mse = mean_squared_error(expected_values, predicted_values, squared=True)
    rmse = mean_squared_error(expected_values, predicted_values, squared=False)
    mae = mean_absolute_error(expected_values, predicted_values)
    r2 = r2_score(expected_values, predicted_values)

    # Print results
    print(f"{name_set} MSE: {mse}; {name_set} RMSE: {rmse}; {name_set} MAE: {mae}; {name_set} R2: {r2}")
    return


def save_and_plot_training_performance(history, save_result, result_path, show_plot, save_plot, plot_path):
    """
    Function to save RMSE performance on training and validation set and plot the curves.
    :param history: History with train and validation performance metrics
    :param save_result: Boolean to save RMSE performance
    :param result_path: String path to save RMSE performance
    :param show_plot: Boolean to show RMSE curves
    :param save_plot: Boolean to save RMSE cuvves
    :param plot_path: String path to save RMSE curves
    """
    # Save RMSE performance
    if save_result:
        result = {"train_rmse": [float(x) for x in history.history["root_mean_squared_error"]],
                  "val_rmse": [float(x) for x in history.history["val_root_mean_squared_error"]]}

        save_path = Path(result_path)
        save_path.mkdir(parents=True, exist_ok=True)
        with open(save_path/"perf_simple_run.json", "w") as fp:
            json.dump(result, fp)

    # Make RMSE curves
    if show_plot or save_plot:
        train_curve = history.history["root_mean_squared_error"]
        val_curve = history.history["val_root_mean_squared_error"]

        plt.plot(train_curve, label="train")
        plt.plot(val_curve, label="validation")
        plt.xlabel("Epochs", fontsize=FONTSIZE_AXES_LABEL)
        plt.ylabel("RMSE", fontsize=FONTSIZE_AXES_LABEL)
        plt.legend(fontsize=FONTSIZE_LEGEND)
        plt.title("Evolution of RMSE on training and validation sets", fontsize=FONTSIZE_TITLE)
        plt.grid()

        # Save plot
        if save_plot:
            save_path = Path(plot_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path/"rmse_train_val.png", dpi=VAL_DPI)
        if show_plot:
            plt.show()
    return


def make_predict_vs_expected_plot(dnn_model, x_test, y_test, norm_outputs, showfig, save_fig, save_path):
    """
    Make plot to compare predicted values with expected values with given model
    :param dnn_model: DNN trained model
    :param x_test: Test set features
    :param y_test: Test set outputs
    :param norm_outputs: Values used to normalize outputs
    :param showfig: Boolean to show the figure
    :param save_fig: Boolean to save the figure
    :param save_path: String path to save the figure
    """
    # Get predictions
    predictions = dnn_model.predict(x=x_test, verbose=0)

    # De-normalize outputs
    predicted_values = predictions * norm_outputs
    expected_values = y_test * norm_outputs

    # Need min and max for the straight line
    min_value = min(min(expected_values), min(predicted_values), 999999)
    max_value = max(max(expected_values), max(predicted_values), -999999)
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
        plt.savefig(save_path / "dnn_predicted_and_observed.png", dpi=VAL_DPI)

    # Show figure
    if showfig:
        plt.show()
    return


def hold_out_evaluation(path_train_set, path_test_set, predicted_expected_plot=True):
    """
    Train a dnn model on glycolysis dataset following a hold out evaluation method.
    :param path_train_set: String path to training set
    :param path_test_set: String path to test set
    :param predicted_expected_plot: Boolean to make predicted vs expected plot
    """
    # Fix seeding
    tf.random.set_seed(123)
    seed(seed=123)
    # Note: despite seeding, results may not be constant when using GPU

    # Check if gpu is available
    try:
        if tf.config.list_physical_devices('GPU'):
            device_name = '/device:GPU:0'
        else:
            device_name = '/device:CPU:0'
    except:
        if tf.test.is_gpu_available():
            device_name = '/device:GPU:0'
        else:
            device_name = '/device:CPU:0'

    # Load training dataset
    train_dataset = pd.read_csv(path_train_set, sep=",")

    # Normalize data and remove id column
    np_train_df = train_dataset.to_numpy()[:, 1:].astype(np.float32)

    max_features = np.max(np_train_df[:, :3])
    max_outputs = np.max(np_train_df[:, 3])

    norm_inputs = np_train_df[:, :3] / max_features
    norm_outputs = np_train_df[:, 3] / max_outputs

    # Create train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(norm_inputs, norm_outputs, test_size=0.2, random_state=123)

    # EarlyStopping function
    early_stopping_fn = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, mode="min", verbose=1,
                                                         restore_best_weights=True)

    # Create the neural network
    kernel = tf.keras.initializers.GlorotUniform(123)  # Initializer for weights matrix
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(3,)),
        tf.keras.layers.Dense(105, activation="elu", kernel_initializer=kernel),
        tf.keras.layers.Dense(105, activation="elu", kernel_initializer=kernel),
        tf.keras.layers.Dense(105, activation="elu", kernel_initializer=kernel),
        tf.keras.layers.Dense(1, activation="sigmoid", kernel_initializer=kernel)], name="Glycolysis_model"
    )

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=[tf.keras.metrics.RootMeanSquaredError(), R2(), tf.keras.metrics.MeanAbsoluteError()]
    )

    # Train model - Verbosity: 0 = silent; 1 = progress bar; 2 = one line per epoch
    train_time = time.time()
    print("Start training...")
    with tf.device(device_name):
        history = model.fit(x_train, y_train, batch_size=100, epochs=3000, validation_data=(x_val, y_val),
                            callbacks=[early_stopping_fn], verbose=1)
    print("----- End of training --------\n")
    train_time = time.time() - train_time
    print(f"Training time: {train_time // 60} min and {train_time % 60} seconds \n")

    # Final evaluation on train and validation sets
    print_performance(x_set=x_train, y_set=y_train, model=model, max_outputs=max_outputs, name_set="Train")
    print_performance(x_set=x_val, y_set=y_val, model=model, max_outputs=max_outputs, name_set="Val")

    # Loading test set
    test_dataset = pd.read_csv(path_test_set, sep=",")
    np_test_df = test_dataset.to_numpy()

    # Remove id and test set name columns
    np_df_test = np_test_df[:, 2:].astype(np.float32)

    # Normalize
    test_inputs = np_df_test[:, :3] / max_features
    test_outputs = np_df_test[:, 3] / max_outputs

    # Evaluating model on test set
    print_performance(x_set=test_inputs, y_set=test_outputs, model=model, max_outputs=max_outputs, name_set="Test")

    # Save and plot training and validation performance
    save_plot_training_perf = False
    if save_plot_training_perf:
        save_and_plot_training_performance(history=history, save_result=True, result_path="results/simple_run",
                                           show_plot=False, save_plot=True, plot_path="plots/simple_run")

    # Make plot that compares expected values and predicted values
    if predicted_expected_plot:
        make_predict_vs_expected_plot(dnn_model=model, x_test=test_inputs, y_test=test_outputs,
                                      norm_outputs=max_outputs, showfig=True, save_fig=True,
                                      save_path="plots/simple_run")
    return


if __name__ == "__main__":
    hold_out_evaluation(path_train_set="data/Table_S1_training_set.csv", path_test_set="data/Table_S2_test_set.csv",
                        predicted_expected_plot=True)
