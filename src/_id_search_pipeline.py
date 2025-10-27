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

def dataset(csv="data/stalcraft_item_names.csv", generated="data/stalcraft_checked.csv"):
    # Load real item IDs
    df_real = pd.read_csv(csv)
    X_real = df_real['id'].tolist()
    y_real = [1] * len(X_real)
    
    ration_1 = sum(y_real) / len(y_real)
    ration_0 = (len(y_real) - sum(y_real)) / len(y_real)
    print(f"Ratio of 1 over rest: {ration_1:.4f}")
    print(f"Ratio of 0 over rest: {ration_0:.4f}")

    # Load generated item IDs
    df_gen = pd.read_csv(generated)
    X_gen = df_gen['id'].tolist()
    y_gen = [0 if pd.isna(hist) else 1 for hist in df_gen['history']]
    
    # Calculate ratios
    ratio_1 = sum(y_gen) / len(y_gen)
    ratio_0 = (len(y_gen) - sum(y_gen)) / len(y_gen)
    print(f"Ratio of 1 over rest: {ratio_1:.4f}")
    print(f"Ratio of 0 over rest: {ratio_0:.4f}")

    # Combine datasets
    X = X_real + X_gen
    y = y_real + y_gen
    
    ratio_0 = (len(y) - sum(y)) / len(y)
    ratio_1 = sum(y) / len(y)
    
    print(f"Combined ratio of 1 over rest: {ratio_1:.4f}")
    print(f"Combined ratio of 0 over rest: {ratio_0:.4f}")
    
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
    
    # print("First 10 ordered IDs:", X_ordered[:10])
    
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
    import pyarrow as pa
    
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

    import pyarrow.parquet as pq

    # Load checked IDs and their info
    checked_df = pd.read_csv("data/stalcraft_checked.csv")
    checked_info = checked_df.set_index("id").to_dict(orient="index")

    # Load item names/types for real items
    names_df = pd.read_csv("data/stalcraft_items_names.csv")
    # names_df = names_df.set_index("id").to_dict(orient="index")

    # Update item_id_db.csv with missing info from names_info and checked_info
    item_db_path = "data/item_id_db.csv"
    item_db = pd.read_csv(item_db_path)

    def get_info(id_):
        name_info = names_df.get(id_, {})
        prob = 1.0 if id_ in names_df else 0.0
        checked = id_ in names_df
        name = name_info.get("name", "")
        type_ = name_info.get("type", "")
        tradable_val = name_info.get("tradable", "")
        tradable = str(tradable_val).lower() == "true"
        history = ""
        return prob, checked, name, type_, tradable, history

    print(get_info("4qdqn"))

    exit(0)

    # Fill missing info for IDs present in names_info
    for idx, row in item_db.iterrows():
        id_ = row["id"]
        if id_ in names_df:
            prob, checked, name, type_, tradable, history = get_info(id_)
            item_db.at[idx, "prob"] = prob
            item_db.at[idx, "checked"] = checked
            item_db.at[idx, "name"] = name
            item_db.at[idx, "type"] = type_
            item_db.at[idx, "tradable"] = tradable
            item_db.at[idx, "history"] = history

    # Save updated CSV
    item_db.to_csv(item_db_path, index=False)

    # # Prepare data for CSV in batches
    # batch_size = 1_000_000
    # csv_path = "data/item_id_db.csv"

    # records = []
    # first_batch = True

    # # Ensure output directory exists
    # os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # for idx, id_ in enumerate(tqdm(all_ids)):
    #     # info = checked_info.get(id_, {})

    #     # prob = 1.0 if id_ in names_info else 0.0
    #     # name = info.get("name", names_info.get(id_, {}).get("name", ""))
    #     # type_ = info.get("type", names_info.get(id_, {}).get("type", ""))
    #     # tradable_val = info.get("tradable", names_info.get(id_, {}).get("tradable", ""))
    #     # tradable = str(tradable_val).lower() == "true"
    #     # history = info.get("history", "")
    #     # checked = id_ in checked_info
        
    #     prob = 0.0
    #     name = ""
    #     type_ = ""
    #     tradable_val = ""
    #     tradable = False
    #     history = ""
    #     checked = False

    #     records.append({
    #         "id": id_,
    #         "prob": prob,
    #         "checked": checked,
    #         "name": name,
    #         "type": type_,
    #         "tradable": tradable,
    #         "history": history
    #     })

    #     # Write batch to CSV
    #     if (idx + 1) % batch_size == 0 or (idx + 1) == len(all_ids):
    #         df = pd.DataFrame(records)
    #         if first_batch:
    #             df.to_csv(csv_path, index=False, mode="w")
    #             first_batch = False
    #         else:
    #             df.to_csv(csv_path, index=False, mode="a", header=False)
    #         records = []


    #     table = pa.Table.from_pandas(pd.DataFrame(records))
    #     pq.write_table(table, "data/item_id_db.parquet")

    # import csv
    # # Prepare data for CSV in batches
    # batch_size = 1_000_000
    # csv_path = "data/item_id_db.csv"

    # os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    # # Open CSV once, write header manually
    # with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["id", "prob", "checked", "name", "type", "tradable", "history"])

    # records = []

    # for idx, id_ in enumerate(tqdm(all_ids)):
    #     # Default (simplified) values
    #     prob = 0.0
    #     name = ""
    #     type_ = ""
    #     tradable = False
    #     history = ""
    #     checked = False

    #     records.append((id_, prob, checked, name, type_, tradable, history))

    #     # Empty list to csv
    #     if (idx + 1) % batch_size == 0 or (idx + 1) == len(all_ids):
    #         with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
    #             writer = csv.writer(f)
    #             writer.writerows(records)
    #         records = []


    
    # batch_size = 10000 # @param batch size

    # csv_path = "data/all_possible_ids_with_prob.csv"
    # header_written = False
    # total_written = 0
    
    # if os.path.exists(csv_path):
    #     # Write the header row manually: id,prob
    #     with open(csv_path, "w") as f:
    #         f.write("id,prob\n")
            
    # print("Starting predictions...")
    
    # top_k = 10_000
    # num_parts = 1
    # part_size = len(all_ids) // num_parts
    # top_k_per_part = top_k // num_parts

    # all_top_ids = []

    # for part in range(num_parts):
    #     start = part * part_size
    #     end = (part + 1) * part_size if part < num_parts - 1 else len(all_ids)
    #     part_ids = all_ids[start:end]

    #     part_probs = []
    #     for i in tqdm(range(0, len(part_ids), batch_size), desc=f"Predicting probabilities (part {part+1}/{num_parts})"):
    #         batch_ids = part_ids[i:i+batch_size]
    #         batch_encoded = np.array([encode_id(x) for x in batch_ids])
    #         batch_probs = clf.predict_proba(batch_encoded)[:, 1]
    #         part_probs.extend(zip(batch_ids, batch_probs))

    #     # Sort and keep top_k_per_part from this part
    #     top_ids_part = sorted(part_probs, key=lambda x: x[1], reverse=True)[:top_k_per_part]
    #     all_top_ids.extend(top_ids_part)

    #     # Write to CSV after each part to avoid storing all in RAM
    #     df_part = pd.DataFrame(top_ids_part, columns=['id', 'prob'])
    #     mode = 'a' if os.path.exists(csv_path) and part > 0 else 'w'
    #     header = not (os.path.exists(csv_path) and part > 0)
    #     df_part.to_csv(csv_path, mode=mode, header=header, index=False)

    # # After all parts, sort the collected top ids and keep top_k overall
    # all_top_ids = sorted(all_top_ids, key=lambda x: x[1], reverse=True)[:top_k]
    # df_top = pd.DataFrame(all_top_ids, columns=['id', 'prob'])
    # df_top.to_csv(csv_path, index=False)
    # print(f"Saved top_k IDs with highest probabilities to {csv_path}")



    
    

