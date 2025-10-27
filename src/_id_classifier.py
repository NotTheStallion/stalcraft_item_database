import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from random import shuffle

# Define possible characters
chars = string.ascii_lowercase + string.digits
char_to_idx = {c: i for i, c in enumerate(chars)}

# Encode ID into numeric vector (one-hot per character)
def encode_id(item_id):
    vec = np.zeros((5, len(chars)))
    for i, ch in enumerate(item_id):
        vec[i, char_to_idx[ch]] = 1
    return vec.flatten()

def combine_valdde_invalde(valid_ids="data/item_id_db.csv", invalid_ids="data/invalid_item_id_db.csv"):
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


def split_dataset(X_ids, y_labels, test_size=0.2):
    X_train_ids, X_test_ids, y_train_labels, y_test_labels = train_test_split(
        X_ids, y_labels, test_size=0.2, random_state=42, stratify=y_labels
    )

    # Encode all IDs
    X = np.array([encode_id(x) for x in X_ids])
    y = np.array(y_labels)

    # Use the previously split train/test sets
    X_train = np.array([encode_id(x) for x in X_train_ids])
    X_test = np.array([encode_id(x) for x in X_test_ids])
    y_train = np.array(y_train_labels)
    y_test = np.array(y_test_labels)
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import mean_squared_error
    from itertools import product
    from tqdm import tqdm
    import os
    from bisect import bisect_left
    import joblib
    
    X_ids, y_labels, X_ordered, _ = dataset()
    X_train, X_test, y_train, y_test = split_dataset(X_ids, y_labels)
    
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_losses = []
    val_losses = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_fold_train, y_fold_train)
        train_loss = np.mean(clf.predict(X_fold_train) != y_fold_train)
        val_loss = np.mean(clf.predict(X_fold_val) != y_fold_val)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

    # Compute test error
    test_preds = clf.predict(X_test)
    test_error = mean_squared_error(y_test, test_preds)
    test_accuracy = np.mean(test_preds == y_test)
    print(f"Test error: {test_error:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")

    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Fold')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Train vs Validation Loss (Error) per Fold')
    plt.savefig("data/id_classifier_loss.png")

    print("Saving model...")
    joblib.dump(clf, "data/id_classifier_model.joblib")


    def is_in_ordered_list(ordered_list, item):
        idx = bisect_left(ordered_list, item)
        return idx < len(ordered_list) and ordered_list[idx] == item

    all_ids = [''.join(p) for p in product(chars, repeat=5)]
    all_ids = [id for id in all_ids if not is_in_ordered_list(X_ordered, id)]
    
    batch_size = 10000 # @param batch size

    csv_path = "data/all_possible_ids_with_prob.csv"
    header_written = False
    total_written = 0
    
    if os.path.exists(csv_path):
        # Write the header row manually: id,prob
        with open(csv_path, "w") as f:
            f.write("id,prob\n")
            
    print("Starting predictions...")
    
    top_k = 10_000
    num_parts = 2
    part_size = len(all_ids) // num_parts
    top_k_per_part = top_k // num_parts

    all_top_ids = []

    for part in range(num_parts):
        start = part * part_size
        end = (part + 1) * part_size if part < num_parts - 1 else len(all_ids)
        part_ids = all_ids[start:end]

        part_probs = []
        for i in tqdm(range(0, len(part_ids), batch_size), desc=f"Predicting probabilities (part {part+1}/{num_parts})"):
            batch_ids = part_ids[i:i+batch_size]
            batch_encoded = np.array([encode_id(x) for x in batch_ids])
            batch_probs = clf.predict_proba(batch_encoded)[:, 1]
            part_probs.extend(zip(batch_ids, batch_probs))

        # Sort and keep top_k_per_part from this part
        top_ids_part = sorted(part_probs, key=lambda x: x[1], reverse=True)[:top_k_per_part]
        all_top_ids.extend(top_ids_part)

        # Write to CSV after each part to avoid storing all in RAM
        df_part = pd.DataFrame(top_ids_part, columns=['id', 'prob'])
        mode = 'a' if os.path.exists(csv_path) and part > 0 else 'w'
        header = not (os.path.exists(csv_path) and part > 0)
        df_part.to_csv(csv_path, mode=mode, header=header, index=False)

    # After all parts, sort the collected top ids and keep top_k overall
    all_top_ids = sorted(all_top_ids, key=lambda x: x[1], reverse=True)[:top_k]
    df_top = pd.DataFrame(all_top_ids, columns=['id', 'prob'])
    df_top.to_csv(csv_path, index=False)
    print(f"Saved top_k IDs with highest probabilities to {csv_path}")



    
    

