# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from naive_bayes import NaiveBayes

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

def get_state_data(df, labels, state):
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
    desired_indices = df["State"] == state

    # extract the reviews and labels from the data dataframe
    desired_reviews = df.loc[desired_indices]
    desired_labels = labels[desired_indices]

    return desired_reviews, desired_labels

def get_five_best_cuisines_for_state(state, state_data, state_predictions):
    """
    Function to get the top five cuisines for the specified state.

    Params:
    state: The desired state to analyze.
    predictions: The predictions made by the model for the specified state.
    """

    cuisines = np.array(['afghan', 'african','american (new)', 'american (traditional)', 'arabian', 'argentine', 'armenian', 'asian fusion', 'australian', 'austrian', 'bangladeshi', 'basque', 'belgian', 'brazilian', 'british', 'bulgarian', 'burmese', 'cajun/creole', 'cambodian', 'caribbean', 'catalan', 'chinese', 'cuban', 'czech', 'eritrean', 'ethiopian', 'filipino', 'french', 'georgian', 'german', 'greek', 'guamanian', 'halal', 'hawaiian', 'himalayan/nepalese', 'honduran', 'hungarian', 'iberian', 'indian', 'indonesian', 'irish', 'italian', 'japanese', 'korean', 'kosher', 'laotian', 'latin american', 'malaysian', 'mediterranean', 'middle eastern', 'mongolian', 'moroccan', 'new mexican cuisine', 'nicaraguan', 'pakistani', 'persian/iranian', 'peruvian', 'polish', 'polynesian', 'portuguese', 'russian', 'scandinavian', 'scottish', 'singaporean', 'slovakian', 'somali', 'spanish', 'sri lankan', 'syrian', 'taiwanese', 'thai', 'turkish', 'ukrainian', 'uzbek', 'vietnamese'])
    num_positive_reviews_per_cuisine = []
    num_negative_reviews_per_cuisine = []

    # iterate over the state data and get the number of positive reviews for each cuisine
    for cuisine in cuisines:
        # get the indices of the reviews/labels which match the current cuisine
        current_cuisine_indices = state_data["Cuisine"] == cuisine

        # get all the labels for the current cuisine
        current_cuisine_labels = state_predictions[current_cuisine_indices]
        
        # get the number of negative/positive reviews
        num_negative_reviews = (current_cuisine_labels == 0).sum()
        num_positive_reviews = (current_cuisine_labels == 1).sum()

        # add the reviews to the respective arrays
        num_negative_reviews_per_cuisine.append(num_negative_reviews)
        num_positive_reviews_per_cuisine.append(num_positive_reviews)
    
    # convert the arrays to numpy arrays
    num_negative_reviews_per_cuisine = np.array(num_negative_reviews_per_cuisine)
    num_positive_reviews_per_cuisine = np.array(num_positive_reviews_per_cuisine)

    # compute the positve:negative ratio for each cuisine
    pos_neg_ratio = num_positive_reviews_per_cuisine / (num_negative_reviews_per_cuisine + 1)

    # sort the ratios -- we only care about the indices -- only want top 5 (descending order) for display
    sorted_ratio_indices = np.argsort(pos_neg_ratio)[::-1][:5]

    # use the sorted indices to get the top 5 cuisines and their ratios
    best_ratios = pos_neg_ratio[sorted_ratio_indices]
    best_cuisines = cuisines[sorted_ratio_indices]

    # return the best cuisines and their respective ratios
    return best_cuisines, best_ratios

def get_best_restaurants_for_state(state_data, state_predictions, best_cuisines_for_state):
    best_restaurants = []

    for cuisine in best_cuisines_for_state:
        # get the indices of the data that match the current cuisine
        cuisine_indices = state_data["Cuisine"] == cuisine

        # get the data and the labels for the cuisine
        restaurant_data = state_data[cuisine_indices]
        labels = state_predictions[cuisine_indices]

        # get all the names of the restaurants
        restaurants = set(restaurant_data["Name"])

        best_ratio = float("-inf")
        best_restaurant = None

        # iterate over all of the businesses
        for restaurant in restaurants:
            # get all the data rows whose review corresponds to the current business
            restaurant_indices = restaurant_data["Name"] == restaurant

            # get the number of positive and negative reviews for the business
            num_positive_reviews = (labels[restaurant_indices] == 1).sum()
            num_negative_reviews = (labels[restaurant_indices] == 0).sum()

            # compute the ratio
            ratio = num_positive_reviews / (num_negative_reviews + 1)

            # check if this ratio is better than our current best
            if ratio > best_ratio:
                best_ratio = ratio
                best_restaurant = restaurant
        
        # add the business to an array
        best_restaurants.append(best_restaurant)

    # return the best restaurants
    return best_restaurants

def graph_five_best_cuisines_for_state(state, best_cuisines, best_ratios):
    """
    Graph the best cuisines ranked by their ratios.

    Params:
    state: The current state.
    best_cuisines: The best cuisines for the specified state.
    best_ratio: The ratio of the cuisines.
    """

    # plot the results
    fig, ax = plt.subplots()

    ax.barh(np.arange(len(best_cuisines)), best_ratios, align="center")
    ax.set_yticks(np.arange(len(best_cuisines)), labels=best_cuisines)
    ax.invert_yaxis()
    ax.set_xlabel("Ratio of Predicted Positive/Negative Reviews")
    ax.set_title(f"Top {len(best_cuisines)} Predicted Best Cuisines in {state.upper()}")

    plt.tight_layout()
    plt.show()

def main():
    # set random seed to guarantee reproducibility
    np.random.seed(0)

    # load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # create our model
    NB = NaiveBayes(alpha=12, vectorize=True)

    # fit the model
    NB.fit(X_train, y_train)

    # states we want to perform analysis on
    results = []
    for state in ["ma", "tx", "wa"]:
        # get the current state data and labels
        current_state_data, current_state_labels = get_state_data(X_test, y_test, state)
        
        # use the model to make predictions for the given state
        current_state_predictions = NB.predict(current_state_data)

        # calculate the accuracy of the model for the given state
        current_state_accuracy = accuracy_score(current_state_labels, current_state_predictions) * 100

        # get the best cuisines for the current states
        best_cuisines_for_current_state, best_ratios_for_current_state = get_five_best_cuisines_for_state(state, current_state_data, current_state_predictions)
        graph_five_best_cuisines_for_state(state, best_cuisines_for_current_state, best_ratios_for_current_state)

        # get the best restaurant for each cuisine for the current state
        best_restaurants_for_current_state = get_best_restaurants_for_state(current_state_data, current_state_predictions, best_cuisines_for_current_state)

        # add the (state, accuracy) tuple to the results array to be displayed later
        results.append([state.upper(), current_state_accuracy, best_cuisines_for_current_state[0].capitalize(), best_restaurants_for_current_state[0]])

    # display the results
    print(tabulate(results, headers=["State", "Sentiment Analysis Accuracy", "Best Cuisine", "Best Restaurant for Cuisine"]))

if __name__ == "__main__":
    main()

