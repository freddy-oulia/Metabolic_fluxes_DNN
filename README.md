# Modeling Metabolic Fluxes Using Deep Learning Based on Enzyme Variations: _Applications to Glycolysis in Entamoeba histolytica_

Metabolic pathway modeling, essential for understanding organism metabolism, is pivotal in predicting genetic mutation effects, drug design, and biofuel development. Enhancing these modeling techniques is crucial for achieving greater prediction accuracy and reliability. However, the limited experimental data or the complexity of the pathway makes it challenging for researchers to predict phenotypes. Deep learning (DL) is known to perform better than other Machine Learning (ML) approaches if the right conditions are met (i.e., a large database and good choice of parameters). Here, we use a knowledge-based model to massively generate synthetic data and extend a small initial dataset of experimental values. The main objective is to assess if DL can perform at least as well as other ML approaches in flux prediction, using 68,950 instances. Two processing methods are used to generate DL models: cross-validation and repeated holdout evaluation. DL models predict the metabolic fluxes with high precision and slightly outperform the best-known ML approach (the Cubist model) with a lower RMSE (≤0.01) in both cases. They also outperform the PLS model (RMSE ≥ 30). This study is the first to use DL to predict the overall flux of a metabolic pathway only from variations of enzyme concentrations.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Running the Scripts](#running-the-scripts)
4. [Project Structure](#project-structure)
5. [References](#references)

## Prerequisites

Before installing this project, make sure you have the following installed on your machine:

- [Python 3.8](https://www.python.org/downloads/)
- [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
- [Git](https://git-scm.com/) (optional, but recommended)

## Environment Setup

Using **Conda**, follow these steps to create the environment with all dependencies:

1. Clone the Git repository:
    ```bash
    git clone https://github.com/freddy-oulia/Metabolic_fluxes_DNN.git
    cd Metabolic_fluxes_DNN
    ```

2. Create a virtual environment:
    ```bash
    conda create -n environment_name python=3.8
    ```

3. Activate the virtual environment:
    ```bash
    conda activate environment_name
    ```

4. Install dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

5. Deactivate the virtual environment:
    ```bash
    conda deactivate
    ```

6. Delete the virtual environment
    ```bash
    conda remove -n environment_name --all
    ```

## Running the Scripts

This project includes several scripts, each designed for a specific task. Below are instructions for running each script.

### 1. Distribution

To see distribution of the dataset, please run the script located at `show_distribution.py` using this:

```bash
python scripts/show_distribution.py
```

### 2. PLS Model

The script for the PLS model is located at `pls_model.py` and is used to train and evaluate the PLS model. We can also make a figure to compare the expected values to the predicted values.
The figure will be in `plots/pls_model/pls_predicted_and observed.png`.
To run the code use this:

```bash
python scripts/pls_model.py
```

### 3. Cubist Model

The script for the Cubist model is located at `cubist_model.py` and is used to train and evaluate the Cubist model. We can also make a figure to compare the expected values to the predicted values.
The figure will be in `plots/cubist_model/cubist_predicted_and observed.png`.
To run the code use this:

```bash
python scripts/cubist_model.py
```

### 4. DNN Model with a simple holdout evaluation

The script for the DNN model is located at `simple_run.py` and is used to train and evaluate the DNN model following a holdout evaluation method.
Results will be saved at `results/simple_run/perf_simple_run.json` and the figure with training and validation RMSE curves can be found at `plots/simple_run/rmse_train_val.png`.
We can also make a figure to compare the expected values to the predicted values found at `plots/simple_run/dnn_predicted_and observed.png`
In the `hold_out_evaluation` method we can change many hyperparameters such as the model architecture in the `number_neurons` variable, the loss function in the `loss_fn` variable, or even the Earlystopping function in the `add_earlystopping` boolean variable.

```bash
python scripts/simple_run.py
```

### 5. DNN Model with repeated holdout evaluation

The script to run the repeated hold out evaluation method with a DNN is found at `repeated_hold_out_evaluation_run.py`.
Results are saved in `results/rho/performances_rho.json`. To make the boxplots or the correlation metrics plots change the `make_boxplot` boolean or the `correlation_metrics` boolean to `True` and make sure that the results file is in the right location.
Plots can be found in `plots/boxplots` and `plots/correlation`. 

```bash
python scripts/repeated_hold_out_evaluation_run.py
```

### 6. DNN Model with repeated cross-validation

The script to run the repeated cross-validation method with a DNN is found at `cross_validation_run.py`.
Results are saved in `results/cv/performances_cv.json`. To make the boxplots or the correlation metrics plots change the `make_boxplot` boolean or the `correlation_metrics` boolean to `True` and make sure that the results file is in the right location.
Plots can be found in `plots/boxplots` and `plots/correlation`. 

```bash
python scripts/cross_validation_run.py
```

### 7. Boxplots comparison

The script to make the boxplots comparison is located at `boxplot_comparison.py`.
The generated figures are saved in `plots/boxplot/comparison/`. Please make sure that the results file for the repeated cross-validation and repeated hold out evaluation are correctly located.

```bash
python scripts/boxplot_comparison.py
```

## Project structure

Here’s an overview of the project structure to help you navigate between the various files and folders:

```bash
Metabolic_fluxes_DNN/
│
├── data/                   # Contains datasets
│   ├── Table_S1_training_set.csv
│   └── Table_S2_test_set.csv
│
├── results/                # Generated  results
│   ├── simple_run
│       └── perf_simple_run.json
│   ├── rho
│       └── performances_rho.json
│   └── cv
│       └── performances_cv.json
│
│
├── plots/                # Generated figures
│   ├── distribution
│       ├── ...
│       └── distribution_Training_set_pgam.png
│   ├── pls_model
│       └── pls_predicted_and observed.png
│   ├── cubist_model
│       └── cubist_predicted_and observed.png
│   ├── simple_run
│       ├── rmse_train_val.png
│       └── dnn_predicted_and observed.png
│   ├── boxplot
│       ├── cross-validation
│            ├── rmse.png
│            ├── mae.png
│            └── r2.png
│       └── repeated_hold_out_evaluation
│            ├── rmse.png
│            ├── mae.png
│            └── r2.png
│   ├── correlation
│       ├── Kendall
│            ├── rmse_cv.png
│            ├── ...
│            └── r2_rho.png
│       ├── Spearman
│            ├── rmse_cv.png
│            ├── ...
│            └── r2_rho.png
│       └── Pearson
│            ├── rmse_cv.png
│            ├── ...
│            └── r2_rho.png
│   └── boxplot_comparison
│       ├── rmse.png
│       ├── mae.png
│       └── r2.png
│
│   # Scripts
├── show_distribution.py
├── pls_model.py
├── cubist_model.py
├── simple_run.py
├── repeated_hold_out_evaluation_run.py
├── cross_validation_run.py
├── boxplot_comparison.py
├── create_dnn_model.py
├── gridearch_run.py
├── metrics.py
├── results_plots.py
│
├── requirements.txt        # Python dependencies for pip
├── README.md               # Project documentation
├── LICENSE-CC-BY.md        # Data license
└── LICENSE                 # Project license
```

## References

Oulia, F.; Charton, P.; Lo-Thong-Viramoutou, O.; Acevedo-Rocha, C.G.; Liu, W.; Huynh, D.; Damour, C.; Wang, J.; Cadet, F. Metabolic Fluxes Using Deep Learning Based on Enzyme Variations: Application to Glycolysis in Entamoeba histolytica. Int. J. Mol. Sci. 2024, 25, 13390. https://doi.org/10.3390/ijms252413390

## License

The complete source code, datasets, and models are available under the Creative Commons Attribution-Non-Commercial ShareAlike International License, Version 4.0 (CC-BY-NC-SA 4.0) for open, non-commercial use.
See the [LICENSE-CC-BY-NC-SA](./LICENSE-BY-NC-SA) file for more details.

[![Creative Commons License](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
