# libraries
import numpy as np
import pandas as pd


def load_data():
    """
    Function which loads the yelp dataset.

    Returns:
    X_train: The training set.
    y_train: The training labels.
    X_val: The validation set.
    y_val: The validation labels.
    X_test: The test set.
    y_test: The test labels.
    """

    X_train = pd.read_csv("../Data/X_train.csv")
    y_train = pd.read_csv("../Data/y_train.csv")["Sentiment"].astype(int).values
    
    X_val = pd.read_csv("../Data/X_val.csv")
    y_val = pd.read_csv("../Data/y_val.csv")["Sentiment"].astype(int).values

    X_test = pd.read_csv("../Data/X_test.csv")
    y_test = pd.read_csv("../Data/y_test.csv")["Sentiment"].astype(int).values

    # return the data
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_staet_data(df, labels, state):
    """
    Function which extracts all those rows whose businesses reside in the given
    city.

    Params:
    df: The data dataframe.
    labels; The labels for the dataframe.
    state: The desired state to filter on.

    Returns:
    data: All the reviews of businesses which reside in the specified state.
    labels: The labels for the data.
    """

    # get all the row indices which match the desired state
    desired_indices = df["State"] == state.upper()

    # extract the reviews and labels from the data dataframe
    desired_reviews = df.loc[desired_indices]
    desired_labels = labels[desired_indices]

    return desired_reviews, desired_labels

def main():
    # set random seed to guarantee reproducibility
    np.random.seed(0)

    # load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    test_cities = set(X_test["City"].values)
    print(sorted(test_cities))

    test_reviews, test_labels = extract_data_given_city(X_test, y_test, "Austin")
    print(len(test_reviews), len(test_labels))

if __name__ == "__main__":
    main()

