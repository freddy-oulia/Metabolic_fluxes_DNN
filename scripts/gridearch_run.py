import json
import tensorflow as tf
import numpy as np
import pandas as pd
from numpy.random import seed
from metrics import R2
from pathlib import Path
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from itertools import combinations
from itertools import product


def create_nn_gridsearch(number_neurons, seed_val=42):
    """
    Create a Deep Neural Network
    :param number_neurons: List with the number of neurons for each layer
    :param seed_val: A seed to create the neural network
    :return: A model
    """
    # Fix seed
    seed(seed_val)
    tf.random.set_seed(seed_val)
    kernel = tf.keras.initializers.GlorotUniform(seed_val)  # Initializer for weights matrix

    # Create the neural network
    layer = tf.keras.layers.Input(shape=(3, ))
    input_layer = layer
    for i in range(len(number_neurons)):
        layer = tf.keras.layers.Dense(number_neurons[i], kernel_initializer=kernel, activation="elu")(layer)
    output = tf.keras.layers.Dense(1, kernel_initializer=kernel, activation="sigmoid")(layer)
    model = tf.keras.models.Model(inputs=input_layer, outputs=output, name="Glycolysis_model")

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.legacy.Adam(),
        metrics=[tf.keras.metrics.RootMeanSquaredError(), R2(), tf.keras.metrics.MeanAbsoluteError()]
    )
    return model


def grid_search_cv(l_hyperparameter, nb_seed=42):
    """
    Grid search for the architecture of the model
    :param l_hyperparameter: Dict containing every hyperparameters needed
    :param nb_seed: Int, seed
    """
    tf.random.set_seed(123)
    np.random.seed(seed=123)

    # Make all architectures possible
    architectures = []
    for i in range(l_hyperparameter["min_hidden_layer"], l_hyperparameter["max_hidden_layer"] + 1):
        combs = combinations(l_hyperparameter["number_neurons"], i)
        for c in combs:
            architectures.append(list(c))

    # Dict to save all res
    all_res = {}

    # Path to save results
    path_res = Path("result_gridsearch")
    path_res.mkdir(parents=True, exist_ok=True)
    path_res = (path_res / "perf_grid_search_cv").with_suffix(".json")

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

    dataset = pd.read_csv("data/Table_S1_training_set.csv", sep=";")

    print("Pre-processing dataset...")
    # Normalize data
    np_df = dataset.to_numpy()
    max_features = np.max(np_df[:, :3])
    max_outputs = np.max(np_df[:, 3])

    inputs = np_df[:, :3] / max_features
    outputs = np_df[:, 3] / max_outputs

    x_train, x_val, y_train, y_val = train_test_split(inputs, outputs, test_size=0.2)

    # EarlyStopping function
    patience = 100
    early_stopping_fn = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=1,
                                                         restore_best_weights=True)

    # Convert continuous target into multi-class to use StratifiedKFold (and generate equivalent folds)
    nb_continuous_class = 4
    y_copy = pd.qcut(y_train, nb_continuous_class, labels=False)
    # Define the K-fold Cross Validator
    seed(nb_seed)
    rskfold = sklearn.model_selection.StratifiedKFold(n_splits=5)

    # For each architecture
    print("Start grid search...")
    print("--------------------------------------")
    for i in range(len(architectures)):
        print(f"\n\tArchitecture: {architectures[i]}")

        current_res = {"architecture": architectures[i],
                       "validation_rmse": []}

        print("Processing folds...")
        # K-fold Cross Validation model evaluation
        for train, validation in rskfold.split(x_train, y_copy):
            # Make a model
            model = create_nn_gridsearch(number_neurons=architectures[i])

            # Fit data to model
            with tf.device(device_name):
                history = model.fit(x_train[train], y_train[train],
                                    batch_size=l_hyperparameter["batch_size"], epochs=l_hyperparameter["epochs"],
                                    validation_data=(x_train[validation], y_train[validation]),
                                    callbacks=[early_stopping_fn], verbose=0)

            # Get performance on test set
            res = model.predict(x=x_val, verbose=0)

            predicted_values = res * max_outputs
            expected_values = y_val * max_outputs
            val_rmse = mean_squared_error(expected_values, predicted_values, squared=False)
            current_res["validation_rmse"].append(float(val_rmse))

            print(f"Validation RMSE on current fold: {val_rmse}\n")

        print("- - - - - - - - - - - - - - - - - - - -")
        # Computing mean on all folds
        current_res["validation_rmse_mean"] = float(np.mean(current_res["validation_rmse"]))
        all_res[i] = current_res

        # Saving results in json file every 10 architectures
        if i % 10 == 0:
            with open(path_res, "w") as fp:
                json.dump(all_res, fp, indent=4)

    # Force save at the end of grid-search
    with open(path_res, "w") as fp:
        json.dump(all_res, fp, indent=4)


def grid_search_normal(l_hyperparameter):
    """
    Grid search for the architecture of the model
    :param l_hyperparameter: Dict containing every hyperparameters needed
    """
    tf.random.set_seed(123)
    np.random.seed(seed=123)

    # Make all architectures possible
    architectures = []
    for i in range(l_hyperparameter["min_hidden_layer"], l_hyperparameter["max_hidden_layer"] + 1):
        combs = product(l_hyperparameter["number_neurons"], repeat=i)
        for c in combs:
            architectures.append(list(c))

    # Dict to save all res
    all_res = {}

    # Path to save results
    path_res = Path("result_gridsearch")
    path_res.mkdir(parents=True, exist_ok=True)
    path_res = (path_res / "perf_grid_search").with_suffix(".json")

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

    dataset = pd.read_csv("data/Table_S1_training_set.csv", sep=";")

    print("Pre-processing dataset...")
    # Normalize data
    np_df = dataset.to_numpy()
    max_features = np.max(np_df[:, :3])
    max_outputs = np.max(np_df[:, 3])

    inputs = np_df[:, :3] / max_features
    outputs = np_df[:, 3] / max_outputs

    x_train, x_val, y_train, y_val = train_test_split(inputs, outputs, test_size=0.2)

    # EarlyStopping function
    patience = 100
    early_stopping_fn = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, mode="min", verbose=1,
                                                         restore_best_weights=True)

    # For each architecture
    print("Start grid search...")
    print("--------------------------------------")
    for i in range(len(architectures)):
        print(f"\nArchitecture: {architectures[i]}")

        model = create_nn_gridsearch(number_neurons=architectures[i])

        # Fit data to model
        with tf.device(device_name):
            _ = model.fit(x_train, y_train, batch_size=l_hyperparameter["batch_size"],
                          epochs=l_hyperparameter["epochs"], validation_data=(x_val, y_val),
                          callbacks=[early_stopping_fn], verbose=0)

        # Get performance on test set
        prediction = model.predict(x=x_val, verbose=0)

        # De-normalize
        predicted_values = prediction * max_outputs
        expected_values = y_val * max_outputs
        val_rmse = mean_squared_error(expected_values, predicted_values, squared=False)
        print(f"RMSE val: {val_rmse}\n")

        # Save result
        all_res[i] = {"architecture": architectures[i], "validation_rmse": float(val_rmse)}

        # Saving results in json file every 10 architectures
        if i % 10 == 0:
            with open(path_res, "w") as fp:
                json.dump(all_res, fp, indent=4)

    # Save at the end of grid-search
    with open(path_res, "w") as fp:
        json.dump(all_res, fp, indent=4)


if __name__ == "__main__":
    make_grid_search_cv = False
    if make_grid_search_cv:
        # Hyperparameters
        hyperparameter = {"min_hidden_layer": 2,
                          "max_hidden_layer": 4,
                          "number_neurons": [i for i in range(5, 151, 5)],
                          "epochs": 1000,
                          "batch_size": 1000}
        grid_search_cv(l_hyperparameter=hyperparameter)

    make_grid_search_normal = False
    if make_grid_search_normal:
        # Hyperparameters
        hyperparameter = {"min_hidden_layer": 3,
                          "max_hidden_layer": 3,
                          "number_neurons": list(range(85, 125, 5)),
                          "epochs": 1000,
                          "batch_size": 1000}
        grid_search_normal(l_hyperparameter=hyperparameter)
