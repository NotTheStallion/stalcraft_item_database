import string
import numpy as np
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
from bisect import bisect_left
from itertools import product


# Define possible characters
CHARS = string.ascii_lowercase + string.digits
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}


# Load HF dataset
def load_hf_dataset(BASE_DIR):
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("hf://datasets/NotTheStallion/stalcraft-database/data/item_id_db.zip")
    csv_path = data_dir / "item_id_db.csv"
    df.to_csv(csv_path, index=False)


# Encode ID into numeric vector (one-hot per character)
def encode_id(item_id):
    vec = np.zeros((5, len(CHARS)))
    for i, ch in enumerate(item_id):
        vec[i, CHAR2IDX[ch]] = 1
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
    if test_size == 0:
        X = np.array([encode_id(x) for x in X_ids])
        y = np.array(y_labels)
        return X, np.array([]), y, np.array([])
    
    X_train_ids, X_test_ids, y_train_labels, y_test_labels = train_test_split(
        X_ids, y_labels, test_size=test_size, random_state=2025, stratify=y_labels
    )

    X_train = np.array([encode_id(x) for x in X_train_ids])
    X_test = np.array([encode_id(x) for x in X_test_ids])
    y_train = np.array(y_train_labels)
    y_test = np.array(y_test_labels)
    
    return X_train, X_test, y_train, y_test


def is_in_ordered_list(ordered_list, item):
        idx = bisect_left(ordered_list, item)
        return idx < len(ordered_list) and ordered_list[idx] == item


def candidate_ids(num_checks = 10_000):
    
    k = len(CHARS)
    length = 5
    total = k ** length

    if num_checks >= total:
        all_ids = [''.join(p) for p in product(CHARS, repeat=length)]
    else:
        if num_checks == 1:
            indices = [0]
        else:
            indices = [ (i * (total - 1)) // (num_checks - 1) for i in range(num_checks) ]

        all_ids = []
        for idx in indices:
            rem = idx
            chars = []
            for pos in range(length - 1, -1, -1):
                base = k ** pos
                digit = rem // base
                chars.append(CHARS[digit])
                rem = rem % base
            all_ids.append(''.join(chars))
    
    return all_ids