import string
import numpy as np
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split


# Define possible characters
chars = string.ascii_lowercase + string.digits
char_to_idx = {c: i for i, c in enumerate(chars)}


# Encode ID into numeric vector (one-hot per character)
def encode_id(item_id):
    vec = np.zeros((5, len(chars)))
    for i, ch in enumerate(item_id):
        vec[i, char_to_idx[ch]] = 1
    return vec.flatten()


def combine_valid_invald(valid_ids="data/item_id_db.csv", invalid_ids="data/invalid_item_id_db.csv"):
    # Load valid & invalid IDs
    df_valid = pd.read_csv(valid_ids)
    X_valid = df_valid['id'].tolist()
    y_valid = [1] * len(X_valid)

    df_invalid = pd.read_csv(invalid_ids)
    X_invalid = df_invalid['id'].tolist()
    y_invalid = [0] * len(X_invalid)
    
    # Combine datasets
    X = X_valid + X_invalid
    y = y_valid + y_invalid
    
    combined = list(zip(X, y))
    shuffle(combined)
    X, y = zip(*combined)
    X = list(X)
    y = list(y)
    
    # Sort X and y together by X
    X_y_ordered = sorted(zip(X, y), key=lambda pair: pair[0])
    X_ordered, y_ordered = zip(*X_y_ordered)
    X_ordered = list(X_ordered)
    y_ordered = list(y_ordered)
    
    return X, y, X_ordered, y_ordered



def split_encode_dataset(X_ids, y_labels, test_size=0.2):
    X_train_ids, X_test_ids, y_train_labels, y_test_labels = train_test_split(
        X_ids, y_labels, test_size=0.2, random_state=2025, stratify=y_labels
    )

    X_train = np.array([encode_id(x) for x in X_train_ids])
    X_test = np.array([encode_id(x) for x in X_test_ids])
    y_train = np.array(y_train_labels)
    y_test = np.array(y_test_labels)
    
    return X_train, X_test, y_train, y_test