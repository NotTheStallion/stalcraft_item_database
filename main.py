#!/usr/bin/env python3
"""CLI entrypoint for the stalcraft_item_database project.

This script provides a small argparse-driven CLI that dispatches to the scripts
in the `src/` directory. It does not re-implement the heavy logic; instead it
executes the existing scripts (so they keep their current behavior) and also
exposes a small helper to call `dataset()` when available.

Usage examples:
  python3 main.py run id_classifier -- --model-arg value
  python3 main.py run check_probs
  python3 main.py dataset

If you need more fine-grained control, pass arguments after `--` and they
will be forwarded to the executed script via sys.argv.
"""

import argparse
import runpy
import sys
from pathlib import Path


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
    import pandas as pd

    # Login using e.g. `huggingface-cli login` to access this dataset
    data_dir = BASE_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv("hf://datasets/NotTheStallion/stalcraft-database/data/item_id_db.zip")
    csv_path = data_dir / "item_id_db.csv"
    df.to_csv(csv_path, index=False)
    
    
    
    # Preprocess dataset
    from src.dataset import combine_valid_invald, split_encode_dataset
    
    


    # Train ID classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    
    
    for iter_idx in range(args.num_iters):
        print(f"\n=== Iteration {iter_idx + 1} / {args.num_iters} ===")
        
        X_ids, y_labels, X_ordered, y_ordered = combine_valid_invald()
        X_train, X_test, y_train, y_test = split_encode_dataset(X_ids, y_labels, test_size=0)
        
        print(f"Total samples: {len(X_ids)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2025)
        model.fit(X_train, y_train)
        
        
        from dataset import is_in_ordered_list, ALL_IDS, encode_id
        
        unchecked_ids = [id for id in ALL_IDS if not is_in_ordered_list(X_ordered, id)]
        
        top_k = args.num_searches
        batch_size = 10000
        num_parts = 2
        part_size = len(unchecked_ids) // num_parts
        top_k_per_part = top_k // num_parts

        all_top_ids = []
        
        from tqdm import tqdm
        import numpy as np
        

        for part in range(num_parts):
            start = part * part_size
            end = (part + 1) * part_size if part < num_parts - 1 else len(unchecked_ids)
            part_ids = unchecked_ids[start:end]

            part_probs = []
            for i in tqdm(range(0, len(part_ids), batch_size), desc=f"Predicting probabilities (part {part+1}/{num_parts})"):
                batch_ids = part_ids[i:i+batch_size]
                batch_encoded = np.array([encode_id(x) for x in batch_ids])
                batch_probs = model.predict_proba(batch_encoded)[:, 1]
                part_probs.extend(zip(batch_ids, batch_probs))
        
        
        
        
    
        
    
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

