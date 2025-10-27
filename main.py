import argparse
import numpy as np
from pathlib import Path
from src.dataset import combine_valid_invald, split_encode_dataset, load_hf_dataset, encode_id, candidate_ids
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stalcraft-db", description="CLI to run src scripts and helpers")
    parser.add_argument("--version", action="version", version="stalcraft-db 0.1")
    
    # Training args
    parser.add_argument("--num_iters", type=int, default=2, help="number of train-search iterations to find item IDs")
    parser.add_argument("--num_searches", type=int, default=100, help="number of searches per iteration to find item IDs")
    parser.add_argument("--save_valid_ids", action="store_true", default=False, help="save valid IDs generated during training to disk")
    parser.add_argument(
        "--save_invalid_ids",
        action="store_true",
        default=False,
        help="save invalid IDs generated during training to disk (useful if planning additional iterations)",
    )
    
    # Extra args
    parser.add_argument("--push_to_hf", action="store_true", default=True, help="push the valid item IDs to Hugging Face Hub")
    args = parser.parse_args(argv)
    
    
    
    # Load dataset from HF
    # @note : Login using e.g. `huggingface-cli login` to access this dataset
    if not args.save_valid_ids:
        load_hf_dataset(BASE_DIR)
    
    for iter_idx in range(args.num_iters):
        print(f"\n=== Iteration {iter_idx + 1} / {args.num_iters} ===")
        
        # Preprocess dataset
        X_ids, y_labels, X_ordered, y_ordered = combine_valid_invald()
        X_train, X_test, y_train, y_test = split_encode_dataset(X_ids, y_labels, test_size=0)
        
        model = LogisticRegression(max_iter=100, class_weight="balanced", random_state=2025, penalty="l2", solver="liblinear")
        model.fit(X_train, y_train)
        
        print("Trained model ... OK")
        
        # Candidate IDs for model prediction
        all_ids = candidate_ids(num_checks = args.num_searches * 3)
        
        X_all = np.array([encode_id(id_str) for id_str in all_ids])
        probs = model.predict_proba(X_all)[:, 1]
        
        
        # Select top-k ids to check using Stalcraft API
        top_k_indices = np.argsort(probs)[-args.num_searches:]
        top_k_ids = [all_ids[i] for i in top_k_indices]
        
        print(f"Checking {len(top_k_ids)} candidate IDs using Stalcraft API ...")
        
        # Validate top-k ids using API
        from src.utils.api import is_id_valid
        valid_ids = []
        invalid_ids = []
        
        for item_id in tqdm(top_k_ids):
            if is_id_valid(item_id):
                valid_ids.append(item_id)
            else:
                invalid_ids.append(item_id)
        
        
        # Store IDs to dataset files
        invalid_path = BASE_DIR / "data" / "invalid_item_id_db.csv"
        if args.save_invalid_ids and invalid_ids:
            with open(invalid_path, "a") as f:
                for invalid_id in invalid_ids:
                    f.write(f"{invalid_id},Item Generated,unknown,False\n")
            print(f"Saved {len(invalid_ids)} invalid IDs to {invalid_path}")
        elif not args.save_invalid_ids and iter_idx == args.num_iters - 1:
            with open(invalid_path, "w") as f:
                f.write("id,name,type,tradable\n")
                f.write("inval,Item One,weapon,True\n")  # Ensure at least one entry
        
        valid_path = BASE_DIR / "data" / "item_id_db.csv"
        with open(valid_path, "a") as f:
            for valid_id in valid_ids:
                f.write(f"{valid_id},Item Generated,unknown,True\n")
        
    print("DONE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

