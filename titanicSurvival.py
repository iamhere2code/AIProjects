import numpy as np
import pandas as pd
import os

import tensorflow as tf
import tensorflow_decision_forests as tfdf

print(f"Found TF-DF {tfdf.__version__}")

train_df = pd.read_csv("/Users/alishuf/Downloads/train.csv")
serving_df = pd.read_csv("/Users/alishuf/Downloads/test.csv")

print(train_df.head(10))

# Preprocess: start by making a copy of the dataset
'''def preprocess(df):
    df = df.copy()
    
    # Make sure the names all start the same naming convention
    def normalize_name(x):
        return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])
    
    # Return the ticket number
    def ticket_number(x):
        return x.split(" ")[-1]
        
    def ticket_item(x):
        items = x.split(" ")
        if len(items) == 1:
            return "NONE"
        return "_".join(items[0:-1])
    
    df["Name"] = df["Name"].apply(normalize_name)
    df["Ticket_number"] = df["Ticket"].apply(ticket_number)
    df["Ticket_item"] = df["Ticket"].apply(ticket_item)                     
    return df
    
preprocessed_train_df = preprocess(train_df)
preprocessed_serving_df = preprocess(serving_df)

preprocessed_train_df.head(5)'''

def normalize_name(x):
    return " ".join([v.strip(",()[].\"'") for v in x.split(" ")])

print(normalize_name(train_df))