"""Code and dataset revisions for a training run, checked against git and the DVC pointers.

``code_revision`` is the git commit; a working tree with uncommitted or untracked changes is
refused unless ``allow_dirty`` is set, and then the revision ends in ``-dirty``.
``dataset_revision`` lists each DVC pointer with the hash of the data actually on disk. The hash
is computed the way DVC 3 computes ``hash: md5`` outputs, so a file that differs from its
pointer is caught; with ``allow_dirty`` it is recorded with a ``-dirty`` suffix instead.
"""

import hashlib
import json
import logging
import re
import subprocess
from collections.abc import Sequence
from pathlib import Path

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

COMMIT_SHA = re.compile(r"[0-9a-f]{40}")
DIRTY_SUFFIX = "-dirty"
DVC_DIRECTORY_SUFFIX = ".dir"


class ProvenanceError(RuntimeError):
    """The run's code or data revision cannot be established."""


class DirtyWorkingTreeError(ProvenanceError):
    """The git working tree has changes that no commit records."""


class DatasetMismatchError(ProvenanceError):
    """Data on disk differs from the hash in its DVC pointer."""


def _git(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as error:
        raise ProvenanceError("git is not installed; the code revision is unknown.") from error
    except subprocess.CalledProcessError as error:
        raise ProvenanceError(
            f"'git {' '.join(arguments)}' failed in {repository}: {error.stderr.strip()}",
        ) from error
    return completed.stdout


def code_revision(repository: Path, *, allow_dirty: bool) -> str:
    """Return the HEAD commit SHA, refusing or marking a dirty working tree."""
    sha = _git(repository, "rev-parse", "HEAD").strip()
    if not COMMIT_SHA.fullmatch(sha):
        raise ProvenanceError(f"git returned an unexpected commit id {sha!r}.")
    changes = _git(repository, "status", "--porcelain", "--untracked-files=all").splitlines()
    if not changes:
        return sha
    if not allow_dirty:
        raise DirtyWorkingTreeError(
            f"The working tree has {len(changes)} uncommitted or untracked change(s). Commit "
            "them, or set training.allow_dirty=true to record the revision as dirty.",
        )
    logger.warning("Training on a dirty working tree; recording %s%s.", sha, DIRTY_SUFFIX)
    return sha + DIRTY_SUFFIX


def _md5_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "md5").hexdigest()


def dvc_md5(path: Path) -> str:
    """The DVC 3 ``hash: md5`` of a file, or of a directory with the ``.dir`` suffix."""
    if path.is_file():
        return _md5_file(path)
    if not path.is_dir():
        raise ProvenanceError(
            f"DVC output {path} does not exist; run 'dvc pull' or 'dvc checkout'."
        )
    entries = sorted(
        (
            {"md5": _md5_file(item), "relpath": item.relative_to(path).as_posix()}
            for item in path.rglob("*")
            if item.is_file()
        ),
        key=lambda entry: entry["relpath"],
    )
    listing = json.dumps(entries, sort_keys=True).encode("utf-8")
    return hashlib.md5(listing).hexdigest() + DVC_DIRECTORY_SUFFIX


def read_dvc_pointer(dvc_file: Path) -> tuple[Path, str]:
    """Return the output path and recorded md5 of a single-output ``.dvc`` file."""
    if not dvc_file.is_file():
        raise ProvenanceError(f"DVC pointer {dvc_file} does not exist.")
    pointer = OmegaConf.to_container(OmegaConf.load(dvc_file))
    outs = pointer.get("outs") if isinstance(pointer, dict) else None
    if not isinstance(outs, list) or len(outs) != 1 or not isinstance(outs[0], dict):
        raise ProvenanceError(f"{dvc_file} must describe exactly one output.")
    out = outs[0]
    if out.get("hash") != "md5" or not isinstance(out.get("md5"), str):
        raise ProvenanceError(f"{dvc_file} must record a DVC 3 'hash: md5' output.")
    if not isinstance(out.get("path"), str):
        raise ProvenanceError(f"{dvc_file} has no output path.")
    return dvc_file.parent / out["path"], out["md5"]


def dataset_revision(
    dvc_files: Sequence[Path],
    data_files: Sequence[Path],
    *,
    allow_dirty: bool,
) -> str:
    """Verify the data against its DVC pointers and return ``pointer=hash`` pairs.

    Every file in ``data_files`` must lie inside one of the pointers' outputs, so the revision
    describes all the data the run reads.
    """
    if not dvc_files:
        raise ProvenanceError("No DVC pointers configured; the dataset revision is unknown.")
    outputs: list[Path] = []
    parts: list[str] = []
    mismatched: list[str] = []
    for dvc_file in dvc_files:
        output, recorded = read_dvc_pointer(dvc_file)
        actual = dvc_md5(output)
        outputs.append(output.resolve())
        if actual == recorded:
            parts.append(f"{dvc_file.as_posix()}={actual}")
        else:
            mismatched.append(dvc_file.as_posix())
            parts.append(f"{dvc_file.as_posix()}={actual}{DIRTY_SUFFIX}")
    untracked = [
        path.as_posix()
        for path in data_files
        if not any(path.resolve().is_relative_to(output) for output in outputs)
    ]
    if untracked:
        raise ProvenanceError(f"Data file(s) {untracked} are not covered by {list(dvc_files)}.")
    if mismatched and not allow_dirty:
        raise DatasetMismatchError(
            f"Data on disk differs from {mismatched}. Run 'dvc checkout' to restore it, "
            "'dvc add' to record it, or set training.allow_dirty=true.",
        )
    if mismatched:
        logger.warning("Training on data that differs from %s.", mismatched)
    return ",".join(parts)
