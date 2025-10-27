import csv
import os
import subprocess
import sys
from pathlib import Path
import pytest



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