# libraries
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, get_scheduler
from transformers.optimization import Adafactor, AdafactorSchedule
from datasets import load_metric

class YelpReviewDataset(Dataset):
    """
    Overridden Pytorch dataset.
    """

    def __init__(self, encodings, labels):
        # store the arguments
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def compute_metrics(eval_pred):
    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def load_data(tokenizer):
    """
    Function which loads the yelp dataset.

    Params:
    tokenizer: The tokenizer used to encode the reviews.

    Returns:
    X_train_encodings: The tokenized training documents.
    y_train: The training labels.
    X_val_encodings: The tokenized validation documents.
    y_val: The validation labels.
    X_test_encodings: The tokenized test documents.
    y_test: The test labels.
    """

    X_train = pd.read_csv("../Data/X_train.csv")["Review"].tolist()
    y_train = pd.read_csv("../Data/y_train.csv")["Sentiment"].astype(int).values
    
    X_val = pd.read_csv("../Data/X_val.csv")["Review"].tolist()
    y_val = pd.read_csv("../Data/y_val.csv")["Sentiment"].astype(int).values

    X_test = pd.read_csv("../Data/X_test.csv")["Review"].tolist()
    y_test = pd.read_csv("../Data/y_test.csv")["Sentiment"].astype(int).values

    # encode the reviews
    X_train_encodings = tokenizer(X_train, truncation=True)
    X_val_encodings = tokenizer(X_val, truncation=True)
    X_test_encodings = tokenizer(X_test, truncation=True)

    # return all the data
    return X_train_encodings, y_train, X_val_encodings, y_val, X_test_encodings, y_test

def main():
    # set the random seed to guarantee reproducibility
    np.random.seed(0)

    # define our model
    pretrained_bert = "bert-base-cased"
    bert_tokenizer = AutoTokenizer.from_pretrained(pretrained_bert)

    # load the data
    X_train_encodings, y_train, X_val_encodings, y_val, X_test_encodings, y_test = load_data(bert_tokenizer)

    # create our datasets
    train_dataset = YelpReviewDataset(X_train_encodings, y_train)
    val_dataset = YelpReviewDataset(X_val_encodings, y_val)
    test_dataset = YelpReviewDataset(X_test_encodings, y_test)
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    # define the training arguments for our model
    training_args = TrainingArguments(
            output_dir="./BERT_training",   # output directory
            num_train_epochs=2,             # number of training iterations
            per_device_train_batch_size=8,  # batch size per device during training
            per_device_eval_batch_size=64,  # batch size for evaluation
            evaluation_strategy="epoch",    # evaluation occurs after each epoch
            logging_dir='./BERT_logs',      # directory for storing logs
            logging_strategy="epoch",       # logging occurs after each epoch
            learning_rate=5e-5,             # learning rate
            seed=0                          # seed for reproducibility
    )

    # create the model -- also define the optimizer and scheduler
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_bert, num_labels=2)
    optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=None)
    lr_scheduler = AdafactorSchedule(optimizer)

    trainer = Trainer(
            model=model,                         # the model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=train_dataset,         # training dataset
            eval_dataset=val_dataset,            # evaluation dataset
            compute_metrics=compute_metrics,     # function to be used in evaluation
            tokenizer=bert_tokenizer,            # enable dynamic padding
            optimizers=(optimizer, lr_scheduler) # optimization technique
    )

    # train the model
    trainer.train()

    # predict on the test set
    test_results = trainer.predict(test_dataset)

    # compute the test accuracy and print results
    test_accuracy = (test_results.predictions.argmax(1) == test_results.label_ids).sum() / len(test_dataset)
    print(f"Test Accuracy: {test_accuracy:.3f}")
        
if __name__ == "__main__":
    main()
