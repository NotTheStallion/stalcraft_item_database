import pandas as pd
from tqdm import tqdm
import os

csv_path = "data/all_possible_ids_with_prob.csv"


# Open and display the first n rows of the CSV file
# n = 20  # You can change n to any number of rows you want to view
df_probs = pd.read_csv(csv_path)
print(len(df_probs))
print(df_probs.head(5))

from test import is_id_valid

valid_ids = []
for idx, row in tqdm(df_probs.iterrows(), total=len(df_probs), desc="Checking all IDs"):
    item_id = row['id']
    prob = row['prob']
    if is_id_valid(item_id):
        valid_ids.append((item_id, prob))
        with open("data/stalcraft_checked.csv", "a") as f:
            f.write(f"{item_id},,,,[]\n")
    else:
        # Add invalid ID row to CSV
        with open("data/stalcraft_checked.csv", "a") as f:
            f.write(f"{item_id},,,,\n")