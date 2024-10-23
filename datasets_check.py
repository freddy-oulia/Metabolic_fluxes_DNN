import pandas as pd
from pathlib import Path


if __name__ == "__main__":
    # Load datasets
    train_df = pd.read_csv(Path("data/Table_S1_training_set.csv"), sep=",")
    test_df = pd.read_csv(Path("data/Table_S2_test_set.csv"), sep=",")

    # Features
    features = ["PGAM", "ENO", "PPDK"]

    train_features = train_df[features]
    test_features = test_df[features]

    # Check if there is duplicates
    duplicates = pd.merge(train_features, test_features, how="inner")

    if not duplicates.empty:
        print(f"There are {len(duplicates)} instances that are present in the training set and the testing set")
        print(duplicates)
    else:
        print("There are no duplicate instances in the training and testing set")
