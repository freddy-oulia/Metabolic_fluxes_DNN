import time
import json
import tensorflow as tf
import numpy as np
import pandas as pd
import create_dnn_model
import results_plots
from metrics import R2
from pathlib import Path
from numpy.random import seed
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


NAME_METRICS = ["mse", "rmse", "r2", "mae"]


def load_train_val_test_sets(path_train_set, path_test_set):
    """
    Load and normalize train, validation and test sets, including the outputs. The 5 test sets are concatenating into 1.
    :param path_train_set: Path to training set
    :param path_test_set: Path to test set
    :returns: Sets of features and outputs for train, validation, test sets and values to de-normalize outputs
    """
    # Training and validation set
    train_set = pd.read_csv(path_train_set, sep=",")
    np_df = train_set.to_numpy()[:, 1:].astype(np.float32)
    max_features = np.max(np_df[:, :3])
    max_outputs = np.max(np_df[:, 3])

    # Normalize
    inputs = np_df[:, :3] / max_features
    outputs = np_df[:, 3] / max_outputs

    x_train, x_val, y_train, y_val = train_test_split(inputs, outputs, test_size=0.2)

    # Loading test sets
    test_set = pd.read_csv(path_test_set, sep=",")

    # 5 different test sets
    np_df_test = test_set.to_numpy()[:, 1:]
    xs_test = []
    ys_test = []

    set_names = ["Test set 1", "Test set 2", "Test set 3", "Test set 4", "Test set 5"]
    for i in range(len(set_names)):
        subset = np_df_test[np.where(np_df_test[:, 0] == set_names[i])]
        subset = subset[:, 1:].astype(np.float32)

        xs_test.append(subset[:, :3] / max_features)
        ys_test.append(subset[:, 3] / max_outputs)

    return x_train, x_val, xs_test, y_train, y_val, ys_test, max_outputs


def init_res_metrics_cv(n_repetition, n_fold, nb_test_set):
    """
    Create dict to save performance of cross-validation
    :param n_repetition: int, number of repetition
    :param n_fold: int, number of fold
    :param nb_test_set: int, number of test set created
    :return: Dict with every keys necessary
    """
    res = {}
    if n_repetition == 1:
        for fold in range(n_fold):
            fold_dict = {}
            for i in range(nb_test_set + 2):
                for metrics_name in NAME_METRICS:
                    if i == 0:
                        fold_dict["train_" + metrics_name] = []
                    elif i == 1:
                        fold_dict["val_" + metrics_name] = []
                    elif nb_test_set == 1:
                        fold_dict["test_" + metrics_name] = []
                    else:
                        fold_dict["test_" + str(i - 2) + "_" + metrics_name] = []
            res["fold_" + str(fold)] = fold_dict

    else:
        for repetition in range(n_repetition):
            repetition_dict = {}
            for fold in range(n_fold):
                fold_dict = {}
                for i in range(nb_test_set + 2):
                    for metrics_name in NAME_METRICS:
                        if i == 0:
                            fold_dict["train_" + metrics_name] = []
                        elif i == 1:
                            fold_dict["val_" + metrics_name] = []
                        elif nb_test_set == 1:
                            fold_dict["test_" + metrics_name] = []
                        else:
                            fold_dict["test_" + str(i - 2) + "_" + metrics_name] = []
                repetition_dict["fold_" + str(fold)] = fold_dict
            res["repetition_" + str(repetition)] = repetition_dict
    return res


def val_test_performance(res_dict, model, x_train, y_train, x_validation, y_validation, x_test, y_test, nb_test,
                         norm_outputs, device):
    """
    Evaluate the trained model on the test set(s) and return performance
    :param res_dict: Dict to save results
    :param model: Trained model
    :param x_train: Training set features
    :param y_train: Training set outputs
    :param x_validation: Validation set features
    :param y_validation: Validation set outputs
    :param x_test: Features on the test set
    :param y_test: y_true
    :param nb_test: Int, number of test set
    :param norm_outputs: Float, value used to normalize outputs
    :param device: String, run either on the gpu or cpu
    :return: Dict with performances on train, validation and test sets
    """
    # Evaluate on training set
    training_set = np.concatenate((x_train, y_train[:, np.newaxis]), axis=1)
    predictions = model.predict(x=training_set[:, :3], verbose=0)
    y_true = training_set[:, 3]

    # De-normalize
    train_predictions = predictions * norm_outputs
    train_ground_truth = y_true * norm_outputs

    res_dict["train_mse"] = float(mean_squared_error(train_ground_truth, train_predictions, squared=True))
    res_dict["train_rmse"] = float(mean_squared_error(train_ground_truth, train_predictions, squared=False))
    res_dict["train_r2"] = float(r2_score(train_ground_truth, train_predictions))
    res_dict["train_mae"] = float(mean_absolute_error(train_ground_truth, train_predictions))

    # Evaluate on validation set
    validation_set = np.concatenate((x_validation, y_validation[:, np.newaxis]), axis=1)
    predictions = model.predict(x=validation_set[:, :3], verbose=0)
    y_true = validation_set[:, 3]

    # De-normalize
    val_predictions = predictions * norm_outputs
    val_ground_truth = y_true * norm_outputs

    res_dict["val_mse"] = float(mean_squared_error(val_ground_truth, val_predictions, squared=True))
    res_dict["val_rmse"] = float(mean_squared_error(val_ground_truth, val_predictions, squared=False))
    res_dict["val_r2"] = float(r2_score(val_ground_truth, val_predictions))
    res_dict["val_mae"] = float(mean_absolute_error(val_ground_truth, val_predictions))

    # Evaluate on test set(s)
    if nb_test == 1:
        # A single test set
        test_set = np.concatenate((x_test, y_test[:, np.newaxis]), axis=1)
        predictions = model.predict(x=test_set[:, :3], verbose=0)
        y_true = test_set[:, 3]

        # De-normalize
        test_predictions = predictions * norm_outputs
        test_ground_truth = y_true * norm_outputs

        res_dict["test_mse"] = float(mean_squared_error(test_ground_truth, test_predictions, squared=True))
        res_dict["test_rmse"] = float(mean_squared_error(test_ground_truth, test_predictions, squared=False))
        res_dict["test_r2"] = float(r2_score(test_ground_truth, test_predictions))
        res_dict["test_mae"] = float(mean_absolute_error(test_ground_truth, test_predictions))
    else:
        set_no = 0
        for features, y_true in zip(x_test, y_test):
            with tf.device(device):
                predictions = model.predict(x=features, verbose=0)
                test_result = []

                # De-normalize
                test_predictions = predictions * norm_outputs
                test_ground_truth = y_true * norm_outputs

                test_result.append(float(mean_squared_error(test_ground_truth, test_predictions, squared=True)))  # MSE
                test_result.append(
                    float(mean_squared_error(test_ground_truth, test_predictions, squared=False)))  # RMSE
                test_result.append(float(r2_score(test_ground_truth, test_predictions)))
                test_result.append(float(mean_absolute_error(test_ground_truth, test_predictions)))

                for i in range(len(NAME_METRICS)):
                    res_dict["test_" + str(set_no) + "_" + NAME_METRICS[i]] = test_result[i]
                set_no += 1
    return res_dict


def cross_valid(l_hyperparameter, print_info=False):
    """
    Function to train and evaluate models based on cross-validation
    :param l_hyperparameter: Dict, hyperparameters for the model and training
    :param print_info: Boolean to print more information during training and evaluation
    """
    # Get train/val/test sets
    if print_info:
        print("Loading and normalizing datasets...")

    train_path = Path("data/Table_S1_training_set.csv")
    test_path = Path("data/Table_S2_test_set.csv")
    x_train, x_val, xs_test, y_train, y_val, ys_test, norm_output_values = load_train_val_test_sets(train_path,
                                                                                                    test_path)

    # Use GPU if available
    if tf.config.list_physical_devices('GPU'):
        device_name = '/device:GPU:0'
        if print_info:
            print('GPU device found')
    else:
        device_name = '/device:CPU:0'
        if print_info:
            print('GPU device not found, using CPU')

    # EarlyStopping function
    early_stopping_fn = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=100, mode="min", verbose=1,
                                                         restore_best_weights=True)

    # Convert continuous target into multi-class to use StratifiedKFold (and generate equivalent folds)
    nb_continuous_class = 4
    y_copy = pd.qcut(y_train, nb_continuous_class, labels=False)

    # Define the K-fold Cross Validator
    seed(10)
    rskfold = RepeatedStratifiedKFold(n_splits=l_hyperparameter["number_folds"],
                                      n_repeats=l_hyperparameter["number_repetitions"])
    fold_no = 0
    repeat_no = 0

    # Vars to get results
    res_metrics_cv = init_res_metrics_cv(n_repetition=l_hyperparameter["number_repetitions"],
                                         n_fold=l_hyperparameter["number_folds"],
                                         nb_test_set=l_hyperparameter["number_test_set"])
    res_metrics_cv["architecture"] = l_hyperparameter["neuron_per_layer"]

    # Paths to save json files
    save_result_path = Path("results/cv")
    save_result_path.mkdir(parents=True, exist_ok=True)

    # Time tracking
    total_time = time.time()

    # K-fold Cross Validation model evaluation
    for train, validation in rskfold.split(x_train, y_copy):
        model = create_dnn_model.create_dnn(number_neurons=l_hyperparameter["neuron_per_layer"],
                                            loss_fn=l_hyperparameter["loss_fn"],
                                            l_metrics=l_hyperparameter["l_metrics"])

        # Say when we start a new repeat cycle
        if fold_no == 0:
            if print_info:
                print("\n****************************************************************************")
            print(f"\tRepeat {repeat_no}: ")

        # Generate a print
        if print_info:
            print('------------------------------------------------------------------------')
        print(f'\tTraining for fold {fold_no} ...')
        time_fold = time.time()

        # Fit data to model
        with tf.device(device_name):
            _ = model.fit(x_train[train], y_train[train], batch_size=l_hyperparameter["batch_size"],
                          epochs=l_hyperparameter["epochs"], validation_data=(x_train[validation], y_train[validation]),
                          callbacks=[early_stopping_fn], verbose=0)
        print("\nCalculation time for fold = ", round(time.time() - time_fold, 4), " s\n")

        # Get the current dict to save results
        if l_hyperparameter["number_repetitions"] > 1:
            current_dict = res_metrics_cv["repetition_" + str(repeat_no)]["fold_" + str(fold_no)]
        else:
            current_dict = res_metrics_cv["fold_" + str(fold_no)]

        # Evaluate model on validation set and test set(s) and save results in dict
        current_dict = val_test_performance(res_dict=current_dict, model=model, x_train=x_train[train],
                                            y_train=y_train[train], x_validation=x_train[validation],
                                            y_validation=y_train[validation], x_test=xs_test, y_test=ys_test,
                                            nb_test=l_hyperparameter["number_test_set"],
                                            norm_outputs=norm_output_values, device=device_name)

        if l_hyperparameter["number_repetitions"] > 1:
            res_metrics_cv["repetition_" + str(repeat_no)]["fold_" + str(fold_no)] = current_dict
        else:
            res_metrics_cv["fold_" + str(fold_no)] = current_dict

        # Increase fold number
        if fold_no == l_hyperparameter["number_folds"] - 1:  # we start a new repeat
            fold_no = 0
            repeat_no += 1
        else:
            fold_no += 1

    # Saving json files
    save_result_path = (save_result_path / "performances_cv").with_suffix(".json")
    with open(save_result_path, "w") as fp:
        json.dump(res_metrics_cv, fp, indent=4)

    print(f"\n\nTotal calculation time = {time.strftime('%H:%M:%S', time.gmtime(time.time() - total_time))} ")


if __name__ == "__main__":
    run_cross_validation = True
    if run_cross_validation:
        # Hyperparameters
        neurons_per_layer = [105, 105, 105]
        loss_function = tf.keras.losses.MeanSquaredError()
        list_metrics = [tf.keras.metrics.RootMeanSquaredError(), R2(), tf.keras.metrics.MeanAbsoluteError()]

        hyperparameter = {"number_test_set": 5,
                          "size_test": 0.2,
                          "number_folds": 5,
                          "number_repetitions": 10,
                          "neuron_per_layer": neurons_per_layer,
                          "loss_fn": loss_function,
                          "l_metrics": list_metrics,
                          "epochs": 3000,
                          "batch_size": 100}
        cross_valid(l_hyperparameter=hyperparameter, print_info=True)

    make_boxplot = False
    if make_boxplot:
        result_path_file = Path("results/cv/performances_cv.json")

        # Check if result file exist
        if result_path_file.is_file():
            with open(result_path_file, "r") as fp:
                file = json.load(fp)

            # Make dataframe & boxplots
            plot_metrics = ["rmse", "mae", "r2"]
            results_plots.make_boxplot(file, plot_metrics, "cv")
        else:
            print("Result file not found. File needed to make boxplot")

    correlation_metrics = False
    if correlation_metrics:
        result_path_file = Path("results/cv/performances_cv.json")

        # Check if result file exist
        if result_path_file.is_file():
            with open(result_path_file, "r") as fp:
                file = json.load(fp)

            results_plots.get_tau(file, "cv")
        else:
            print("Result file not found. File needed to make correlation matrix plots")
