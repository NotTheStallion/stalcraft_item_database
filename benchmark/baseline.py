import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
import string
import sys
from pathlib import Path

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.api import is_id_valid
from dataset import encode_id, CHARS, CHAR2IDX
from tqdm import tqdm
import random




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



def main():
    k = 200  # set how many random ids you want
    
    all_ids = generate_random_ids(k)

    valid_ids = []
    for idx, id_str in enumerate(tqdm(all_ids, desc="Checking IDs")):
        if is_id_valid(id_str):
            valid_ids.append(id_str)
        if (idx + 1) % 100 == 0:
            print(f"Checked {idx + 1} IDs, found {len(valid_ids)} valid so far")
    print(f"Number of valid IDs in first 10000: {len(valid_ids)}")




if __name__ == "__main__":
    raise SystemExit(main())

