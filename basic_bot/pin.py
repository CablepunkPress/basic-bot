"""Pin Hugging Face model files for a hardware profile.

    python -m basic_bot.pin REPO FILE [FILE ...] [--revision REV]

Looks up the commit the repository is at now, or the given revision,
and prints, for each file, the profile lines that pin it: a url that
names the commit, the file's sha256, and its size. Paste them into
the model's entry in the profile.

The engine downloads only pinned files, so every profile, official
or custom, needs these lines for each model it downloads.

Example:
    python -m basic_bot.pin bartowski/gemma-4-12B-it-GGUF gemma-4-12B-it-Q6_K_L.gguf
"""

import argparse
import json
import posixpath
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, NoReturn

API = "https://huggingface.co/api/models"
SITE = "https://huggingface.co"


def _fail(message: str) -> NoReturn:
    sys.exit(f"ERROR: {message}")


def _get(url: str) -> Any:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            _fail(f"not found on Hugging Face: {url}")
        _fail(f"Hugging Face returned HTTP {e.code} for {url}")
    except urllib.error.URLError as e:
        _fail(f"could not reach Hugging Face: {e.reason}")


def _commit(repo: str, revision: str) -> str:
    """The full commit a branch, tag, or commit refers to."""
    rev = urllib.parse.quote(revision, safe="")
    return _get(f"{API}/{repo}/revision/{rev}")["sha"]


def _entry(repo: str, commit: str, path: str) -> dict:
    """The repository's listing entry for one file at one commit."""
    folder = posixpath.dirname(path)
    url = f"{API}/{repo}/tree/{commit}"
    if folder:
        url += "/" + urllib.parse.quote(folder)

    for entry in _get(url):
        if entry.get("path") == path and entry.get("type") == "file":
            return entry
    _fail(f"{path} is not in {repo} at commit {commit[:7]}")


def _pin_lines(repo: str, commit: str, path: str) -> str:
    entry = _entry(repo, commit, path)
    lfs = entry.get("lfs")
    if not lfs or "oid" not in lfs:
        _fail(
            f"{path} isn't stored with Git LFS, so Hugging Face publishes "
            f"no sha256 for it. Model files normally are; check the name."
        )

    size = entry.get("size") or lfs.get("size")
    url = f"{SITE}/{repo}/resolve/{commit}/{urllib.parse.quote(path)}"

    return "\n".join([
        f"# {repo} at {commit[:7]}, {size / 1e9:.2f} GB",
        f'file = "{posixpath.basename(path)}"',
        f'url = "{url}"',
        f'sha256 = "{lfs["oid"]}"',
        f"size = {size}",
    ])


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m basic_bot.pin",
        description="Print the profile lines that pin Hugging Face model files.",
    )
    parser.add_argument("repo", help="repository, e.g. bartowski/gemma-4-12B-it-GGUF")
    parser.add_argument("files", nargs="+", help="file names within the repository")
    parser.add_argument(
        "--revision", default="main",
        help="branch, tag, or commit to pin (default: main, as it is now)",
    )
    args = parser.parse_args()

    commit = _commit(args.repo, args.revision)
    print("\n\n".join(_pin_lines(args.repo, commit, f) for f in args.files))


if __name__ == "__main__":
    main()
