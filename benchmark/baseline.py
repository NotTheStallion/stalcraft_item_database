import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
import string
from test import is_id_valid
from tqdm import tqdm

chars = string.ascii_lowercase + string.digits
char_to_idx = {c: i for i, c in enumerate(chars)}
all_ids = [''.join(p) for p in product(chars, repeat=5)]

valid_ids = []
for idx, id_str in enumerate(tqdm(all_ids[:10000], desc="Checking IDs")):
    if is_id_valid(id_str):
        valid_ids.append(id_str)
    if (idx + 1) % 100 == 0:
        print(f"Checked {idx + 1} IDs, found {len(valid_ids)} valid so far")
print(f"Number of valid IDs in first 10000: {len(valid_ids)}")

