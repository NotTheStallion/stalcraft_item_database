import os
import zipfile
from huggingface_hub import HfApi, HfFolder, Repository
from datetime import datetime

# Configurations
csv_file = "data/item_id_db.csv"
zip_file = "data/item_id_db.zip"
repo_id = "NotTheStallion/stalcraft-database"
description = "This dataset contains the IDs of all items in Stalcraft up to date. The CSV is zipped for efficient storage."
card_content = f"""
---
license: mit
task_categories:
- tabular-classification
pretty_name: Stalcraft Item ID Database
dataset_info:
    size: {os.path.getsize(csv_file)} bytes
    date: {datetime.now().strftime('%Y-%m-%d')}
---

# Stalcraft Market Analysis

{description}
"""


# Load HF repo
def load_hf_repo():
    repo_local_dir = "hf_repo"
    repo = Repository(local_dir=repo_local_dir, clone_from=repo_id, repo_type="dataset")
    return repo_local_dir

# Zip updated CSV file from data directory
def zip_dataset(repo_local_dir):
    zip_file = os.path.join(repo_local_dir, "data", "item_id_db.zip")
    csv_file = os.path.join("data", "item_id_db.csv")
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_file, arcname=os.path.basename(csv_file))
    return zip_file


# push hf_repo to hub
def push_to_hf(repo_local_dir):
    repo = Repository(local_dir=repo_local_dir, clone_from=repo_id, repo_type="dataset")
    repo.push_to_hub(commit_message="Update zipped CSV dataset")


if __name__ == "__main__":
    repo_local_dir = load_hf_repo()
    zip_file = zip_dataset(repo_local_dir)
    push_to_hf(repo_local_dir)
    
    
