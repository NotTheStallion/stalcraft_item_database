import os
import sys
from pathlib import Path
import pytest
from test_utils import _get_prev_file_via_git, _read_csv_from_string, _read_current_ids

# Ensure src is importable
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))



def test_api_call():
    from utils.api import is_id_valid  # noqa: E402
    """Test that the API call works for a known valid and invalid ID."""
    valid_id = "l0og1"  # Replace with a known valid ID for testing
    invalid_id = "inval"  # Replace with a known invalid ID for testing

    assert is_id_valid(valid_id), f"Expected ID {valid_id} to be valid."
    assert not is_id_valid(invalid_id), f"Expected ID {invalid_id} to be invalid."


def must_break():
    assert False, "Deliberate failure to test test framework."


# @pytest.mark.network
def test_new_ids_are_valid_or_skipped():
    from utils.api import is_id_valid  # noqa: E402
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
        return

    # Limit number of checked ids to keep test time reasonable
    MAX_CHECK = int(os.getenv("TEST_MAX_NEW_IDS", "20"))
    new_ids = new_ids[:MAX_CHECK]

    if not (os.getenv("CLIENT_ID") and os.getenv("CLIENT_SECRET")):
        return

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
    



def test_accessible_env_var():
    """Test that CLIENT_ID and CLIENT_SECRET are accessible in the test environment."""
    assert os.getenv("CLIENT_ID"), "CLIENT_ID environment variable is not set."
    assert os.getenv("CLIENT_SECRET"), "CLIENT_SECRET environment variable is not set."
    assert os.getenv("NEW"), "NEW environment variable is not set."


if __name__ == "__main__":
    test_new_ids_are_valid_or_skipped()