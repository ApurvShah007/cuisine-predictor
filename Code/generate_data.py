import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def read_data(path):
    json_data = []
    with open(path) as f:
        for line in f:
            json_data.append(json.loads(line))

    return json_data

def get_cuisine(categories):
    yelp_cuisine_categories = ['afghan', 'african','american (new)', 'american (traditional)', 'arabian', 'argentine', 'armenian', 'asian fusion', 'australian', 'austrian', 'bangladeshi', 'basque', 'belgian', 'brazilian', 'british', 'bulgarian', 'burmese', 'cajun/creole', 'cambodian', 'caribbean', 'catalan', 'chinese', 'cuban', 'czech', 'eritrean', 'ethiopian', 'filipino', 'french', 'georgian', 'german', 'greek', 'guamanian', 'halal', 'hawaiian', 'himalayan/nepalese', 'honduran', 'hungarian', 'iberian', 'indian', 'indonesian', 'irish', 'italian', 'japanese', 'korean', 'kosher', 'laotian', 'latin american', 'malaysian', 'mediterranean', 'middle eastern', 'mongolian', 'moroccan', 'new mexican cuisine', 'nicaraguan', 'pakistani', 'persian/iranian', 'peruvian', 'polish', 'polynesian', 'portuguese', 'russian', 'scandinavian', 'scottish', 'singaporean', 'slovakian', 'somali', 'spanish', 'sri lankan', 'syrian', 'taiwanese', 'thai', 'turkish', 'ukrainian', 'uzbek', 'vietnamese']

    return next((category for category in categories if category in yelp_cuisine_categories), None)

def create_data(review_path, business_path="../Data/yelp_academic_dataset_business.json"):
    # load the business and the reviews at the given file locations
    business_data = read_data(business_path)
    review_data = read_data(review_path)

    # variables to hold the final data
    X = [] # will hold the data
    y = [] # will hold the labels

    # iterate over the businesses
    # need to extract business_id, city, name, stars
    business_dict = {}
    for business_obj in business_data:
        # get the ID, city, and name of business
        business_id = business_obj["business_id"] # get the business ID
        business_city = business_obj["city"].lower().strip() # get the city of the business -- lowercase it
        business_state = business_obj["state"].lower().strip() # get the state of the business -- lowercase it
        business_name = business_obj["name"] # get the name of the business
        if business_obj["categories"] is None:
            business_categories = None
        else:
            business_categories = [category.lower().strip() for category in business_obj["categories"].split(",")]

        # create business object using id as key
        business_dict[business_id] = {
                "city": business_city, 
                "state": business_state, 
                "name": business_name, 
                "categories": business_categories
        }

    # iterate over the reviews
    for review_object in review_data:
        # get the review_id business_id, review_text, and stars
        review_id = review_object["review_id"]                      # get the review ID
        business_id = review_object["business_id"]                  # get the ID of the business being reviewed
        review_text = review_object["text"].lower().strip()         # get the review itself
        review_year = review_object["date"][:4]                     # get the year of the review
        stars = review_object["stars"]                              # get the star rating of the review

        # if the business being reviewed is not a restaurant, then ignore it
        if business_dict[business_id]["categories"] is None or "restaurants" not in business_dict[business_id]["categories"]:
            continue

        # get the type of cuisine
        cuisine = get_cuisine(business_dict[business_id]["categories"])
        # if the cuisine is none, then ignore it
        if cuisine is None:
            continue

        # create data from all the information
        data_sample = [
                business_id,
                business_dict[business_id]["name"],
                business_dict[business_id]["city"],
                business_dict[business_id]["state"],
                cuisine,
                review_id,
                review_year,
                review_text
        ]

        # append data_sample and label to respective arrays
        X.append(data_sample)
        y.append(1 if stars >= 3 else 0)

    # return review_dict, review_labels
    return X, y

def main():
    # determine the file path for the necessary files
    train_review_path = "../Data/training_reviews.json"
    test_review_path = "../Data/test_reviews.json"
    output_dir = "../Data"

    # now extract the training and test data using method above
    training_data, training_labels = create_data(train_review_path)
    X_test, y_test = create_data(test_review_path)

    # use sklearn's train-test-split to split training data into training and validation sets
    # use 70-30 train/val split
    X_train, X_val, y_train, y_val = train_test_split(training_data, training_labels, test_size=0.3, random_state=0)

    # create the dataframes for each set
    X_train_df = pd.DataFrame(X_train)
    y_train_df = pd.DataFrame(y_train)

    X_val_df = pd.DataFrame(X_val)
    y_val_df = pd.DataFrame(y_val)

    X_test_df = pd.DataFrame(X_test)
    y_test_df = pd.DataFrame(y_test)


    # write out each dataframe to csv
    data_headers = ["BusinessId", "Name", "City", "State", "Cuisine", "ReviewId", "Year", "Review"]
    label_headers = ["Sentiment"]

    X_train_df.to_csv(f"{output_dir}/X_train.csv", sep=",", header=data_headers, index=False, encoding="utf-8", chunksize=1024)
    y_train_df.to_csv(f"{output_dir}/y_train.csv", sep=",", header=label_headers, index=False, encoding="utf-8", chunksize=1024)

    X_val_df.to_csv(f"{output_dir}/X_val.csv", sep=",", header=data_headers, index=False, encoding="utf-8", chunksize=1024)
    y_val_df.to_csv(f"{output_dir}/y_val.csv", sep=",", header=label_headers, index=False, encoding="utf-8", chunksize=1024)

    X_test_df.to_csv(f"{output_dir}/X_test.csv", sep=",", header=data_headers, index=False, encoding="utf-8", chunksize=1024)
    y_test_df.to_csv(f"{output_dir}/y_test.csv", sep=",", header=label_headers, index=False, encoding="utf-8", chunksize=1024)

if __name__ == "__main__":
    main()
