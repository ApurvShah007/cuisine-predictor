import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression

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

    X_train = pd.read_csv("../Data/X_train.csv")["Review"].values
    y_train = pd.read_csv("../Data/y_train.csv")["Sentiment"].astype(int).values
    
    X_val = pd.read_csv("../Data/X_val.csv")["Review"].values
    y_val = pd.read_csv("../Data/y_val.csv")["Sentiment"].astype(int).values

    X_test = pd.read_csv("../Data/X_test.csv")["Review"].values
    y_test = pd.read_csv("../Data/y_test.csv")["Sentiment"].astype(int).values

    # create our count vectorizer and transform our data
    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # return the data
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    # set the random seed to guarantee reproducibility
    np.random.seed(0)

    # load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # perform hyperparameter tuning
    C = np.arange(1, 6, 1)
    best_c = -1
    best_accuracy = float("-inf")
    validation_accuracies = []

    for c in C:
        # create the model -- use l2 regularization
        LR = LogisticRegression(C = c, tol = 0.01, max_iter = 1000, solver = "sag")

        # fit the model
        LR.fit(X_train, y_train)

        # make predictions using the validation set
        y_val_pred = LR.predict(X_val)

        # compute the accuracy
        validation_accuracy = accuracy_score(y_val, y_val_pred)

        # check if validation accuracy is better than previous
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_c = c

        # add the validation accuracy to the list of accuracies
        validation_accuracies.append(validation_accuracy)

    # using the best hyperparameter value c, make predictions on the test set
    LR = LogisticRegression(C = best_c)
    LR.fit(X_train, y_train)
    y_test_pred = LR.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    
    # print the accuracy and the confusion matrix
    print(f"Testing Accuracy for regularization value (C) {best_c}: {test_accuracy * 100}%")
    print(f"Confusion Matrix:\n\tTP: {tp} FP: {fp}\n\tFN: {tn} TN: {tn}")

    plt.plot(C, validation_accuracies, color="#d13636")
    plt.xlabel("Amount of Regularization (C)")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. Amount of Regularization (C)")
    plt.show()

if __name__ == "__main__":
    main()
