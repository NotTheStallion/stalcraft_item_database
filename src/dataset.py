import string
import numpy as np
import pandas as pd
from random import shuffle
from sklearn.model_selection import train_test_split
from bisect import bisect_left
import random


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


def generate_random_ids(k, length=5, seed=42):
    max_combos = len(CHARS) ** length
    if k > max_combos:
        raise ValueError(f"k ({k}) is larger than the number of possible combos ({max_combos})")
    if seed is not None:
        random.seed(seed)
    ids = set()
    while len(ids) < k:
        ids.add(''.join(random.choices(CHARS, k=length)))
    return list(ids)



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


def candidate_ids(num_checks=10_000, existing_ids=None):
    if existing_ids is None:
        existing_ids = []

    # helper conversions
    k = len(CHARS)
    length = 5
    total = k ** length

    def id_to_idx(item_id):
        idx = 0
        for ch in item_id:
            if ch not in CHAR2IDX:
                return None
            idx = idx * k + CHAR2IDX[ch]
        return idx

    def idx_to_id(idx):
        rem = idx
        chars = []
        for pos in range(length - 1, -1, -1):
            base = k ** pos
            digit = rem // base
            chars.append(CHARS[digit])
            rem = rem % base
        return ''.join(chars)

    # build set of existing indices (only valid ids counted)
    existing_idx = set()
    for eid in existing_ids:
        if isinstance(eid, str) and len(eid) == length:
            eidx = id_to_idx(eid)
            if eidx is not None:
                existing_idx.add(eidx)

    available = total - len(existing_idx)
    if available <= 0:
        return []

    # If asking for all (or more than available), return all remaining ids
    if num_checks >= available:
        all_ids = []
        for idx in range(total):
            if idx not in existing_idx:
                all_ids.append(idx_to_id(idx))
        return all_ids

    # choose evenly spaced base indices
    if num_checks == 1:
        base_indices = [0]
    else:
        base_indices = [ (i * (total - 1)) // (num_checks - 1) for i in range(num_checks) ]

    chosen_idx = []
    chosen_set = set()

    # for each base index, probe forward to find a non-existing, non-duplicate index
    for b in base_indices:
        idx = b
        # attempt up to total times (should normally find quickly)
        for _ in range(total):
            if idx not in existing_idx and idx not in chosen_set:
                chosen_idx.append(idx)
                chosen_set.add(idx)
                break
            idx = (idx + 1) % total
        if len(chosen_idx) >= num_checks:
            break

    # if we still need more (due to collisions), scan sequentially
    idx = 0
    while len(chosen_idx) < num_checks:
        if idx not in existing_idx and idx not in chosen_set:
            chosen_idx.append(idx)
            chosen_set.add(idx)
        idx += 1
        if idx >= total:
            break

    all_ids = [idx_to_id(i) for i in chosen_idx]
    return all_ids