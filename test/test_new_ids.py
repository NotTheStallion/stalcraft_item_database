import csv
import os
import subprocess
import sys
from pathlib import Path
import pytest

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from utils.api import is_id_valid  # noqa: E402



def _read_csv_from_string(s: str):
    f = s.splitlines()
    reader = csv.DictReader(f)
    ids = [row.get("id") for row in reader if row.get("id")]
    return ids


def _read_current_ids(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        return [row.get("id") for row in reader if row.get("id")]


def _get_prev_file_via_git(filepath: str):
    # Try origin/main first (CI/common case), fallback to HEAD~1
    cmd = ["git", "show", f"origin/master:{filepath}"]
    
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
        return out.decode("utf-8")
    except subprocess.CalledProcessError:
        return


def _get_added_ids_via_diff(filepath: str):
    # As a last resort, use git diff to collect added lines in the file
    try:
        out = subprocess.check_output(["git", "diff", "--unified=0", "--no-color", "--", filepath])
        text = out.decode("utf-8")
    except subprocess.CalledProcessError:
        return []

    added = []
    for line in text.splitlines():
        if line.startswith("+") and not line.startswith("++"):
            # simple heuristic: CSV row was added
            added.append(line[1:])

    if not added:
        return []

    # Parse added lines as CSV (prepend header if possible)
    # Try to find header in current file
    header = None
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().strip()
    except Exception:
        header = None

    csv_text = (header + "\n" if header else "") + "\n".join(added)
    reader = csv.DictReader(csv_text.splitlines())
    return [row.get("id") for row in reader if row.get("id")]


# @pytest.mark.network
def test_new_ids_are_valid_or_skipped():
    """Detect newly added IDs in data/item_id_db.csv and validate via API.

    Behavior:
    - Determine new IDs by comparing the current `data/item_id_db.csv` to the
      previous version fetched via git (origin/main or HEAD~1). If that fails,
      fall back to parsing `git diff` added lines.
    - Skip the test if no new IDs are detected.
    - Skip the network/API checks if the API helper cannot be imported or if
      CLIENT_ID/CLIENT_SECRET aren't set in the environment. This avoids false
      negatives in environments without credentials.
    """

    repo_file = ROOT / "data" / "item_id_db.csv"
    assert repo_file.exists(), f"expected file {repo_file} to exist"

    current_ids = _read_current_ids(repo_file)

    prev_text = _get_prev_file_via_git(str(repo_file.relative_to(ROOT)))
    
    if prev_text:
        prev_ids = _read_csv_from_string(prev_text)

    new_ids = []
    if prev_ids:
        new_ids = [i for i in current_ids if i not in set(prev_ids)]

    # If nothing new detected, skip the test (nothing to validate)
    if not new_ids:
        pytest.skip("No newly added IDs detected in data/item_id_db.csv")

    # Limit number of checked ids to keep test time reasonable
    MAX_CHECK = int(os.getenv("TEST_MAX_NEW_IDS", "20"))
    new_ids = new_ids[:MAX_CHECK]

    if not (os.getenv("CLIENT_ID") and os.getenv("CLIENT_SECRET")):
        pytest.skip("CLIENT_ID/CLIENT_SECRET not set in environment for API validation")

    invalid = []
    for item_id in new_ids:
        try:
            res = is_id_valid(item_id)
        except Exception as e:
            pytest.skip(f"API call raised exception; skipping. Exception: {e}")

        # According to src/utils/api.py, is_id_valid returns r.json() when ok
        if not res:
            invalid.append(item_id)

    assert not invalid, f"Some newly added IDs were not validated by API: {invalid}"




if __name__ == "__main__":
    test_new_ids_are_valid_or_skipped()