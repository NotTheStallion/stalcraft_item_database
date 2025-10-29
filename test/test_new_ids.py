import os
import sys
from pathlib import Path
import pytest
import subprocess
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


# def test_must_break():
#     assert False==True, "Deliberate failure to test test framework."


# @pytest.mark.network
def test_new_ids_are_valid_or_skipped():
    from utils.api import is_id_valid  # noqa: E402
    """Detect newly added IDs by comparing current branch dataset to master and validate via API.

    Steps:
    - Load the dataset from the current branch (working tree).
    - Fetch and load the dataset from `origin/master` (or `origin/main`).
    - Compute newly added IDs (present in current, not in master).
    - Validate those new IDs via the API and fail if any are invalid.
    """

    repo_file = ROOT / "data" / "item_id_db.csv"
    assert repo_file.exists(), f"expected file {repo_file} to exist"

    # Load current branch dataset (PR / working tree)
    current_ids = _read_current_ids(repo_file)

    # Try to fetch master from origin to ensure origin/master is available
    try:
        subprocess.check_call(["git", "fetch", "origin", "master"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # ignore fetch failure; next step will try multiple refs
        pass

    prev_text = None
    for ref in ("origin/master", "origin/main", "master", "main", "HEAD~1"):
        try:
            out = subprocess.check_output(["git", "show", f"{ref}:{str(repo_file.relative_to(ROOT))}"], stderr=subprocess.DEVNULL)
            prev_text = out.decode("utf-8")
            break
        except subprocess.CalledProcessError:
            continue

    if prev_text is None:
        # As a last resort, try the helper which checks a few common specs
        prev_text = _get_prev_file_via_git(str(repo_file.relative_to(ROOT)))

    if prev_text is None:
        pytest.skip("Could not retrieve master/base version of data/item_id_db.csv to compare")

    prev_ids = _read_csv_from_string(prev_text)

    # Compute new IDs (present in current but not in prev/master)
    prev_set = set(prev_ids)
    new_ids = [i for i in current_ids if i not in prev_set]

    if not new_ids:
        pytest.skip("No newly added IDs detected in data/item_id_db.csv")

    # Limit number of checked ids to keep test time reasonable
    MAX_CHECK = int(os.getenv("TEST_MAX_NEW_IDS", "100"))
    new_ids = new_ids[:MAX_CHECK]

    # Require credentials to run validation (CI should supply them)
    if not (os.getenv("CLIENT_ID") and os.getenv("CLIENT_SECRET")):
        pytest.skip("CLIENT_ID/CLIENT_SECRET not set in environment for API validation")

    invalid = []
    for item_id in new_ids:
        try:
            res = is_id_valid(item_id)
        except Exception as e:
            pytest.skip(f"API call raised exception; skipping. Exception: {e}")

        # is_id_valid returns a JSON-like truthy object when valid, otherwise None/False
        if not res:
            invalid.append(item_id)

    assert not invalid, f"Some newly added IDs were not validated by API: {invalid}"
    



def test_accessible_env_var():
    """Test that CLIENT_ID and CLIENT_SECRET are accessible in the test environment."""
    assert os.getenv("CLIENT_ID"), "CLIENT_ID environment variable is not set."
    assert os.getenv("CLIENT_SECRET"), "CLIENT_SECRET environment variable is not set."
    assert os.getenv("NEW"), "NEW environment variable is not set."
