import json
import numpy as np
import pandas as pd
import codecs
from pandas.io.formats.format import TextAdjustment
from sklearn.model_selection import train_test_split
import NB_classifier

PATH_TO_BUSINESS = "../Data/yelp_dataset/yelp_academic_dataset_business.json"
PATH_TO_REVIEW = "../Data//yelp_dataset/yelp_academic_dataset_review_trunc.json"

def read_data(path):
    json_data = []
    with open(path) as f:
        for line in f:
            json_data.append(json.loads(line))

    return json_data

def create_data():
    business_data = read_data(PATH_TO_BUSINESS)
    review_data = read_data(PATH_TO_REVIEW)
    X = []
    y = []

    # need to extract business_id, city, name, stars
    business_dict = {}
    for business_obj in business_data:
        # get the ID, city, and name of business
        business_id = business_obj["business_id"]
        business_city = business_obj["city"]
        business_name = business_obj["name"]
        # create business object using id as key
        business_dict[business_id] = {"city": business_city, "name": business_name}


    for review_object in review_data:
        # get the review_id business_id, review_text, and stars
        review_id = review_object["review_id"]
        business_id = review_object["business_id"]
        review_text = review_object["text"]
        stars = review_object["stars"]

        # create data from this
        data_sample = [business_id,
                       business_dict[business_id]["name"],
                       business_dict[business_id]["city"],
                       review_id,
                       review_text
                       ]

        label = 1 if stars >= 3 else 0

        # append data_sample and label to respective arrays
        X.append(data_sample)
        y.append(label)

    # return review_dict, review_labels
    return X, y

# extract the data and remove any null rows
X, y = create_data()

# perform train/val/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.30, random_state=42)

# write the data to csv file for later use
data_headers = ["BusinessID", "Name", "City", "ReviewID", "Review"]
label_headers = ["Sentiment"]
pd.DataFrame(X_train).to_csv("../data/yelp_dataset/X_train.csv", sep = ",", header = data_headers, index = False, encoding = "utf-8", chunksize = 1024)
pd.DataFrame(y_train).to_csv("../data/yelp_dataset/y_train.csv", sep = ",", header = label_headers, index = False, encoding = "utf-8", chunksize = 1024)

pd.DataFrame(X_val).to_csv("../data/yelp_dataset/X_val.csv", sep = ",", header = data_headers, index = False, encoding = "utf-8", chunksize = 1024)
pd.DataFrame(y_val).to_csv("../data/yelp_dataset/y_val.csv", sep = ",", header = label_headers, index = False, encoding = "utf-8", chunksize = 1024)

pd.DataFrame(X_test).to_csv("../data/yelp_dataset/X_test.csv", sep = ",", header = data_headers, index = False, encoding = "utf-8", chunksize = 1024)
pd.DataFrame(y_test).to_csv("../data/yelp_dataset/y_test.csv", sep = ",", header = label_headers, index = False, encoding = "utf-8", chunksize = 1024)