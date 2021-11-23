import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

class NaiveBayes():
    def  __init__(self, alpha):
        # hyperparameter for laplace smoothing -- avoid zero frequency problem
        # cannot be negative
        self.alpha = 1 if alpha <= 0 else alpha

        self.POS_LABEL = 1
        self.NEG_LABEL = 0

        # variables for class probabilities
        self.log_class_probabilities = {self.NEG_LABEL: 0,
                                        self.POS_LABEL: 0
                                       }

        self.log_review_probabilities = {self.NEG_LABEL: None,
                                         self.POS_LABEL: None
                                        }

    def fit(self, X_train, y_train):
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
        # calculate the posterior probability for the negative class
        # negative_posterior = np.dot(self.log_review_probabilities[self.NEG_LABEL], test_review) + self.log_class_probabilities[self.NEG_LABEL]
        negative_posterior = (test_review @ self.log_review_probabilities[self.NEG_LABEL].T) + self.log_class_probabilities[self.NEG_LABEL]
        positive_posterior = (test_review @ self.log_review_probabilities[self.POS_LABEL].T) + self.log_class_probabilities[self.POS_LABEL]

        return 1 if positive_posterior > negative_posterior else 0

    def predict(self, X_test):
        y_pred = np.array([self.__predict(test_review) for test_review in X_test])
        return y_pred

def load_data():
    X_train = pd.read_csv("../Data/yelp_dataset/X_train.csv")["Review"].values
    y_train = pd.read_csv("../Data/yelp_dataset/y_train.csv")["Sentiment"].astype(int).values

    X_val = pd.read_csv("../Data/yelp_dataset/X_val.csv")["Review"].values
    y_val = pd.read_csv("../Data/yelp_dataset/y_val.csv")["Sentiment"].astype(int).values

    X_test = pd.read_csv("../Data/yelp_dataset/X_test.csv")["Review"].values
    y_test = pd.read_csv("../Data/yelp_dataset/y_test.csv")["Sentiment"].astype(int).values

    vectorizer = CountVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_val = vectorizer.transform(X_val)
    X_test = vectorizer.transform(X_test)

    # return X_train, y_train, X_test, y_test
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    alphas = np.arange(1, 25, 1)
    best_alpha = -1
    best_accuracy = float("-inf")
    accuracies = []

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
        accuracies.append(validation_accuracy)

    # using the best alpha found, get the accuracy on the test set
    NB = NaiveBayes(best_alpha)
    NB.fit(X_train, y_train)
    y_test_pred = NB.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Testing Accuracy for alpha value {best_alpha}: {test_accuracy * 100}%")

    plt.plot(alphas, accuracies)
    plt.xlabel("Alpha")
    plt.ylabel("Validation Accuracy")
    plt.title("Validation Accuracy vs. Alpha")
    plt.show()

if __name__ == "__main__":
    main()
