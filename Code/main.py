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

def get_cuisine_trends(state_data, state_predictions):
    """
    Gets the predicted best cuisine type over the years of 2011 to 2021 
    for the current state.

    Params:
    state_data: The dataframe containing information on the desired state.
    state_predictions: The predicted sentiments on the restaurants in state_data.

    Returns:
    best_cuisine_per_year: An array of the form [(year, best_cuisine, sentiment_score)]
    where the year corresponds to the year, best_cuisine corresponds to the best
    cuisine for that year, and sentiment_score is the score that was given to
    the best_cuisine.
    """

    years = [year for year in range(2021-10, 2022)]
    cuisines = np.array(['afghan', 'african','american (new)', 'american (traditional)', 'arabian', 'argentine', 'armenian', 'asian fusion', 'australian', 'austrian', 'bangladeshi', 'basque', 'belgian', 'brazilian', 'british', 'bulgarian', 'burmese', 'cajun/creole', 'cambodian', 'caribbean', 'catalan', 'chinese', 'cuban', 'czech', 'eritrean', 'ethiopian', 'filipino', 'french', 'georgian', 'german', 'greek', 'guamanian', 'halal', 'hawaiian', 'himalayan/nepalese', 'honduran', 'hungarian', 'iberian', 'indian', 'indonesian', 'irish', 'italian', 'japanese', 'korean', 'kosher', 'laotian', 'latin american', 'malaysian', 'mediterranean', 'middle eastern', 'mongolian', 'moroccan', 'new mexican cuisine', 'nicaraguan', 'pakistani', 'persian/iranian', 'peruvian', 'polish', 'polynesian', 'portuguese', 'russian', 'scandinavian', 'scottish', 'singaporean', 'slovakian', 'somali', 'spanish', 'sri lankan', 'syrian', 'taiwanese', 'thai', 'turkish', 'ukrainian', 'uzbek', 'vietnamese'])


    best_cuisine_per_year = []

    for year in years:
        # get the data and labels from the current year
        year_data, year_labels = state_data[state_data["Year"] == year], state_predictions[state_data["Year"] == year]

        # iterate over the cuisines
        best_cuisine = None
        best_sentiment_score = float("-inf")
        for cuisine in cuisines:
            # get the data and labels from the current cuisine
            cuisine_data, cuisine_labels = year_data[year_data["Cuisine"] == cuisine], year_labels[year_data["Cuisine"] == cuisine]

            # compute the number of positive and negative reviews for this cuisine type
            num_positive_reviews = (cuisine_labels == 1).sum()
            num_negative_reviews = (cuisine_labels == 0).sum()

            # compute the ratio
            sentiment_score = num_positive_reviews / (num_negative_reviews + 1)

            # check if we can update the best_cuisine variable
            if sentiment_score > best_sentiment_score:
                best_sentiment_score = sentiment_score
                best_cuisine = cuisine
        
        best_cuisine_per_year.append((year, best_cuisine, best_sentiment_score))
    
    return best_cuisine_per_year

def graph_cuisine_trends(state, trends):
    """
    Graph the given historic trends for the specified state.

    Params:
    state: The state to display trends for.
    trends: See the return of get_cuisine_trends.
    """

    x = [trend[0] for trend in trends] # the years
    labels = [trend[1] for trend in trends] # get the type of cuisine
    height = [trend[2] for trend in trends] # the sentiment score
    width = 0.8 # width of the bars

    fig, ax = plt.subplots()

    font = {
            "fontweight" : "bold",
            "fontsize"   : 6}

    bar_chart = ax.bar(np.arange(len(x)), height, align="center", width=width)

    plt.title(f"Most Popular Predicted Cuisine by Year for {state.upper()}", pad=20)
    plt.xlabel("Year")
    plt.xticks(np.arange(len(x)), x)
    plt.ylabel("Sentiment Score")

    # attach the cuisine type to each bar
    for i, bar in enumerate(bar_chart):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., 1.05*height,
                "%s" % labels[i],
                ha="center", va="bottom", **font)
    
    plt.show()

def get_best_overall_cuisine(state_data, state_predictions):
    """
    Function which gets the best cuisine and its sentiment score over all of
    the available years.

    Params:
    state_data: The dataframe containing information on the state.
    state_predictions: The predicted sentiment on the business in the state_data.

    Returns:
    best_cuisine: The overall best cuisine over all the available data for the state.
    best_sentiment_score: The sentiment score for the best_cuisine.
    """

    # get list of cuisines
    cuisines = np.array(['afghan', 'african','american (new)', 'american (traditional)', 'arabian', 'argentine', 'armenian', 'asian fusion', 'australian', 'austrian', 'bangladeshi', 'basque', 'belgian', 'brazilian', 'british', 'bulgarian', 'burmese', 'cajun/creole', 'cambodian', 'caribbean', 'catalan', 'chinese', 'cuban', 'czech', 'eritrean', 'ethiopian', 'filipino', 'french', 'georgian', 'german', 'greek', 'guamanian', 'halal', 'hawaiian', 'himalayan/nepalese', 'honduran', 'hungarian', 'iberian', 'indian', 'indonesian', 'irish', 'italian', 'japanese', 'korean', 'kosher', 'laotian', 'latin american', 'malaysian', 'mediterranean', 'middle eastern', 'mongolian', 'moroccan', 'new mexican cuisine', 'nicaraguan', 'pakistani', 'persian/iranian', 'peruvian', 'polish', 'polynesian', 'portuguese', 'russian', 'scandinavian', 'scottish', 'singaporean', 'slovakian', 'somali', 'spanish', 'sri lankan', 'syrian', 'taiwanese', 'thai', 'turkish', 'ukrainian', 'uzbek', 'vietnamese'])

    best_cuisine = None
    best_sentiment_score = float("-inf")

    # iterate over the each cuisine and get the one with the highest sentiment score
    for cuisine in cuisines:
        # get the predictions for the current cuisine
        cuisine_labels = state_predictions[state_data["Cuisine"] == cuisine]
        
        # get number of positive and negative reviews
        num_positive_reviews = (cuisine_labels == 1).sum()
        num_negative_reviews = (cuisine_labels == 0).sum()

        # compute sentiment score
        sentiment_score = num_positive_reviews / (num_negative_reviews + 1)

        # check if sentiment score is better than current best
        if sentiment_score > best_sentiment_score:
            best_sentiment_score = sentiment_score
            best_cuisine = cuisine

    return best_cuisine, best_sentiment_score

def get_best_restaurant(state_data, state_predictions, best_overall_cuisine):
    """
    Function which returns the best restaurant that serves the specified best
    overall cuisine for the given state.

    Params:
    state_data: The dataframe containing information on the state.
    state_predictions: The predicted sentiment on the business in the state_data.
    best_overall_cuisine: The cuisine to find the best restaurant for.

    Returns:
    best_restaurant: The name of the best restaurant which serves the specified cuisine.
    best_sentiment_score: The sentiment score for the best_restaurant.
    """

    # get the cuisine data and labels for the 
    cuisine_data = state_data[state_data["Cuisine"] == best_overall_cuisine]
    cuisine_labels = state_predictions[state_data["Cuisine"] == best_overall_cuisine]

    best_restaurant = None
    best_sentiment_score = float("-inf")

    restaurants = set(cuisine_data["BusinessId"]) # get the set of all available restaurants
    for restaurant in restaurants:
        # get the data and the labels for the current restaurant
        restaurant_data = cuisine_data[cuisine_data["BusinessId"] == restaurant]
        restaurant_labels = cuisine_labels[cuisine_data["BusinessId"] == restaurant]
        restaurant_name = restaurant_data.loc[restaurant_data["BusinessId"] == restaurant, "Name"].iloc[0] # get the name of restaurant

        # get the number of positive and negative reviews
        num_positive_reviews = (restaurant_labels == 1).sum()
        num_negative_reviews = (restaurant_labels == 0).sum()

        # compute the sentiment score
        sentiment_score = num_positive_reviews / (num_negative_reviews + 1)

        # check if sentiment score is better than current best
        if sentiment_score > best_sentiment_score:
            best_sentiment_score = sentiment_score
            best_restaurant = restaurant_name

    return best_restaurant, best_sentiment_score

def main():
    # set random seed to guarantee reproducibility
    np.random.seed(0)

    # load the data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()

    # create our model
    NB = NaiveBayes(alpha=4, vectorize=True)

    # fit the model
    NB.fit(X_train, y_train)

    # iterate over all the interested states
    results = []
    for state in ["ma", "tx", "wa"]:
        # get the data and labels for the corresponding state
        state_data = X_test[X_test["State"] == state]
        # make predictions on the state data
        state_predictions = NB.predict(state_data)

        # get the cuisine trends over the last 10 years and graph the results
        trends = get_cuisine_trends(state_data, state_predictions)
        #graph_cuisine_trends(state, trends)

        # get the overall cuisine
        best_overall_cuisine, best_overall_cuisine_sentiment_score = get_best_overall_cuisine(state_data, state_predictions)

        # get the best restaurant for the best overall cuisine
        best_restaurant, best_restaurant_sentiment_score = get_best_restaurant(state_data, state_predictions, best_overall_cuisine)

        # add these results to table
        results.append([
            state.upper(),
            best_overall_cuisine.capitalize(),
            round(best_overall_cuisine_sentiment_score),
            best_restaurant,
            round(best_restaurant_sentiment_score)
        ])

    # tabulate the results
    print(tabulate(results, headers=["State", "Best Overall Cuisine", "Cuisine Sentiment Score", "Best Restaurant for Cuisine", "Restaurant Sentiment Score"]))

if __name__ == "__main__":
    main()
