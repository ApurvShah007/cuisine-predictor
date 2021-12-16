import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from nltk.corpus import stopwords

class NaiveBayes():
    def  __init__(self, alpha, vectorize=False):
        # hyperparameter for laplace smoothing -- avoid zero frequency problem
        # cannot be negative
        self.alpha = 1 if alpha <= 0 else alpha
        self.vectorize = vectorize
        self.vectorizer = CountVectorizer(stop_words=stopwords.words("english")) if vectorize else None

        self.POS_LABEL = 1
        self.NEG_LABEL = 0

        # variables for class probabilities
        self.log_class_probabilities = {self.NEG_LABEL: 0,
                                        self.POS_LABEL: 0
                                       }

        # variables for the probability of each word given the review label 
        self.log_review_probabilities = {self.NEG_LABEL: None,
                                         self.POS_LABEL: None
                                        }

    def fit(self, X_train, y_train):
        """
        Function which "trains" the model given the training data. In essence,
        this function calculates the log-class-probabilities and the log-probability
        for each word.

        Params:
        X_train: The training data.
        y_train: The training labels.
        """

        # check if we need to vectorize
        if self.vectorize:
            X_train = self.vectorizer.fit_transform(X_train["Review"])

        # compute the class probabilities
        self.log_class_probabilities[self.NEG_LABEL] = np.log(np.sum(y_train == self.NEG_LABEL) / len(y_train))
        self.log_class_probabilities[self.POS_LABEL] = np.log(np.sum(y_train == self.POS_LABEL) / len(y_train))

        # get all the negative/positive reviews
        negative_reviews = X_train[y_train == self.NEG_LABEL, :]
        positive_reviews = X_train[y_train == self.POS_LABEL, :]

        # get the total word counts for the negative/positive reviews
        # add alpha for laplace smoothing
        negative_review_word_counts = np.sum(negative_reviews, axis = 0) + self.alpha
        positive_review_word_counts = np.sum(positive_reviews, axis = 0) + self.alpha

        # calculate the evidence probabilities
        self.log_review_probabilities[self.NEG_LABEL] = np.log(negative_review_word_counts / np.sum(negative_review_word_counts))
        self.log_review_probabilities[self.POS_LABEL] = np.log(positive_review_word_counts / np.sum(positive_review_word_counts))

    def __predict(self, test_review):
        """
        Helper function which predicts the label for a single test instance by
        calculating and comparing the unnormalized posterior probabilities of
        the given instance.

        Params:
        test_reivew: The instance to make a prediction for.

        Returns:
        1 if the test_review belongs to the positive class, 0 otherwise.
        """

        # calculate the posterior probability for the negative class
        negative_posterior = (test_review @ self.log_review_probabilities[self.NEG_LABEL].T) + self.log_class_probabilities[self.NEG_LABEL]
        positive_posterior = (test_review @ self.log_review_probabilities[self.POS_LABEL].T) + self.log_class_probabilities[self.POS_LABEL]

        return 1 if positive_posterior > negative_posterior else 0

    def predict(self, X_test):
        """
        Function which generates a list of predictions for each test review in
        the specified test set.

        Params:
        X_test: The test set to make predictions for.

        Returns:
        y_pred: The predictions for the specified test set.
        """

        # check if we need to vectorize
        if self.vectorize:
            X_test = self.vectorizer.transform(X_test["Review"])

        y_pred = np.array([self.__predict(test_review) for test_review in X_test])
        return y_pred

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
    vectorizer = CountVectorizer(stop_words=stopwords.words("english"))
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # return the data
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # perform hyperparameter tuning using grid search
    alphas = np.arange(1, 21, 1)
    best_alpha = -1
    best_accuracy = float("-inf")
    validation_accuracies = []

    for alpha in alphas:
        # create our model
        NB = NaiveBayes(alpha)
        # fit the model on the training data
        NB.fit(X_train, y_train)
        # make predictions on the test data
        y_pred = NB.predict(X_val)

        # compute the accuracy
        validation_accuracy = accuracy_score(y_val, y_pred)

        # check if this accuracy is better than what we've seen before
        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_alpha = alpha

        # append this accuracy to the list
        validation_accuracies.append(validation_accuracy)

    # using the best alpha found, get the accuracy on the test set and confusion matrix
    NB = NaiveBayes(best_alpha)
    NB.fit(X_train, y_train)
    y_test_pred = NB.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    
    # print the accuracy and the confusion matrix
    print(f"Testing Accuracy for alpha value {best_alpha}: {test_accuracy * 100}%")
    print(f"Confusion Matrix:\n\tTP: {tp} FP: {fp}\n\tFN: {fn} TN: {tn}")

    plt.plot(alphas, validation_accuracies, color="#d13636")
    plt.xlabel("Alpha")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. Alpha")
    plt.show()

if __name__ == "__main__":
    main()
