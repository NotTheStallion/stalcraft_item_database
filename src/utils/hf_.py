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

# Zip the CSV file
with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_file, arcname=os.path.basename(csv_file))

# Authenticate with Hugging Face
token = HfFolder.get_token()
api = HfApi()

# Create repo if it doesn't exist
try:
        api.create_repo(repo_id, repo_type="dataset", exist_ok=True)
except Exception:
        pass

# Clone repo locally
repo_local_dir = "hf_repo"
repo = Repository(local_dir=repo_local_dir, clone_from=repo_id, repo_type="dataset", use_auth_token=token)

# Copy files to repo
os.makedirs(repo_local_dir, exist_ok=True)
target_data_dir = os.path.join(repo_local_dir, "data")
os.makedirs(target_data_dir, exist_ok=True)
if not os.path.exists(zip_file):
    raise FileNotFoundError(f"{zip_file} does not exist. Please check the path and try again.")

os.replace(zip_file, os.path.join(target_data_dir, os.path.basename(zip_file)))
with open(os.path.join(repo_local_dir, "README.md"), "w") as f:
        f.write(card_content)

# Push to Hugging Face Hub
repo.push_to_hub(commit_message="Add zipped CSV dataset and database card")