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
    parser.add_argument("--nums_searches", type=int, default=1000, help="number of searches per iteration to find item IDs")
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
    
    X_ids, y_labels, X_ordered, y_ordered = combine_valid_invald()
    X_train, X_test, y_train, y_test = split_encode_dataset(X_ids, y_labels)
    
    print(f"Total samples: {len(X_ids)}")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    train_ratio = sum(y_train) / len(y_train)
    test_ratio = sum(y_test) / len(y_test)
    print(f"Training valid ID ratio: {train_ratio:.4f}")
    print(f"Testing valid ID ratio: {test_ratio:.4f}")


    # Train ID classifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import log_loss
    
    
    for iter_idx in range(args.num_iters):
        print(f"\n=== Iteration {iter_idx + 1} / {args.num_iters} ===")

        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=2025)
        train_losses = []
        val_losses = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            clf = LogisticRegression(max_iter=1000)
            clf.fit(X_fold_train, y_fold_train)

            train_proba = clf.predict_proba(X_fold_train)
            val_proba = clf.predict_proba(X_fold_val)
            train_losses.append(log_loss(y_fold_train, train_proba))
            val_losses.append(log_loss(y_fold_val, val_proba))

        mean_train_loss = sum(train_losses) / len(train_losses)
        mean_val_loss = sum(val_losses) / len(val_losses)
        print(f"CV mean train loss: {mean_train_loss:.4f}, CV mean val loss: {mean_val_loss:.4f}")

        # Retrain on full training set and evaluate on test set
        model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=2025)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        print("\nClassification Report on Test Set:")
        print(classification_report(y_test, y_pred, digits=4))
        
        
    
        
    
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

