import csv
import subprocess
from pathlib import Path



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
    # Try the previous commit on remote master first (CI/common case),
    # then local master~1, then HEAD~1 as a last resort.
    candidates = [
        f"origin/master~1:{filepath}",
        f"master~1:{filepath}",
        f"HEAD~1:{filepath}",
    ]

    for spec in candidates:
        cmd = ["git", "show", spec]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            return out.decode("utf-8")
        except subprocess.CalledProcessError:
            continue

    return None