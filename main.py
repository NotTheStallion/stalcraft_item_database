import argparse
import numpy as np
from pathlib import Path
from src.dataset import combine_valid_invald, split_encode_dataset, load_hf_dataset, CHARS, encode_id
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from itertools import product
import random


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stalcraft-db", description="CLI to run src scripts and helpers")
    parser.add_argument("--version", action="version", version="stalcraft-db 0.1")
    
    # Training args
    parser.add_argument("--num_iters", type=int, default=2, help="number of train-search iterations to find item IDs")
    parser.add_argument("--num_searches", type=int, default=1000, help="number of searches per iteration to find item IDs")
    parser.add_argument(
        "--save-invalid-ids",
        action="store_true",
        default=True,
        help="save invalid IDs generated during training to disk (useful if planning additional iterations)",
    )
    
    # Extra args
    parser.add_argument("--push-to-hf", action="store_true", default=True, help="push the valid item IDs to Hugging Face Hub")
    args = parser.parse_args(argv)
    
    
    
    # Load dataset from HF
    # @note : Login using e.g. `huggingface-cli login` to access this dataset
    load_hf_dataset(BASE_DIR)
    
    for iter_idx in range(args.num_iters):
        print(f"\n=== Iteration {iter_idx + 1} / {args.num_iters} ===")
        
        # Preprocess dataset
        X_ids, y_labels, X_ordered, y_ordered = combine_valid_invald()
        X_train, X_test, y_train, y_test = split_encode_dataset(X_ids, y_labels, test_size=0)
        
        model = LogisticRegression(max_iter=100, class_weight="balanced", random_state=2025, penalty="l2", solver="liblinear")
        model.fit(X_train, y_train)
        
        print("Trained model ... OK")
        
        
        
        N = args.num_searches * 3
        # genrate N equally separated ids of length 5 from CHARS
        
        
        k = len(CHARS)
        length = 5
        total = k ** length

        if N >= total:
            all_ids = [''.join(p) for p in product(CHARS, repeat=length)]
        else:
            if N == 1:
                indices = [0]
            else:
                indices = [ (i * (total - 1)) // (N - 1) for i in range(N) ]

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
        
        X_all = np.array([encode_id(id_str) for id_str in all_ids])
        probs = model.predict_proba(X_all)[:, 1]
        
        # Select top-k ids
        top_k_indices = np.argsort(probs)[-args.num_searches:]
        top_k_ids = [all_ids[i] for i in top_k_indices]
        
        
        
        
        
        # Validate top-k ids
        from src.utils.api import is_id_valid
        valid_ids = []
        invalid_ids = []
        
        for item_id in top_k_ids:
            if is_id_valid(item_id):
                valid_ids.append(item_id)
            else:
                invalid_ids.append(item_id)
        
        
        
        
        
        

        
        
        
        
    
        
    
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

