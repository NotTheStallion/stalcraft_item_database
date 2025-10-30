import sys
from pathlib import Path
from tqdm import tqdm

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.api import is_id_valid  # noqa: E402
from dataset import  generate_random_ids  # noqa: E402





def main():
    k = 1000  # set how many random ids you want
    
    all_ids = generate_random_ids(k)

    valid_ids = []
    for idx, id_str in enumerate(tqdm(all_ids, desc="Checking IDs")):
        if is_id_valid(id_str):
            valid_ids.append(id_str)
        if (idx + 1) % 100 == 0:
            print(f"Checked {idx + 1} IDs, found {len(valid_ids)} valid so far")
    print(f"Number of valid IDs : {len(valid_ids)}")




if __name__ == "__main__":
    raise SystemExit(main())

