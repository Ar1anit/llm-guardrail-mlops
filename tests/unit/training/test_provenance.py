"""Code and dataset revisions: git state and DVC pointers, checked on real files."""

from pathlib import Path

import pytest
from src.training.provenance import (
    DatasetMismatchError,
    DirtyWorkingTreeError,
    ProvenanceError,
    code_revision,
    dataset_revision,
    dvc_md5,
    read_dvc_pointer,
)

from tests.unit.training.support import init_repository, write_dvc_pointer

# Hashes written by `dvc add` (DVC 3.67) for the files created in `_write_dvc_fixture`.
DVC_DIRECTORY_MD5 = "45ee81697ccaba6ad37d92dc4864a830.dir"
DVC_FILE_MD5 = "9799de695214694b45480490979bd18d"


def _write_dvc_fixture(root: Path) -> tuple[Path, Path]:
    (root / "data" / "nested").mkdir(parents=True)
    (root / "data" / "one.jsonl").write_text('{"text": "a"}\n', encoding="utf-8")
    (root / "data" / "nested" / "two.txt").write_text("second file\n", encoding="utf-8")
    (root / "golden.jsonl").write_text("golden\n", encoding="utf-8")
    return (
        write_dvc_pointer(root / "data.dvc", "data", DVC_DIRECTORY_MD5),
        write_dvc_pointer(root / "golden.jsonl.dvc", "golden.jsonl", DVC_FILE_MD5),
    )


def test_clean_tree_yields_the_head_commit(tmp_path: Path) -> None:
    sha = init_repository(tmp_path)

    assert code_revision(tmp_path, allow_dirty=False) == sha


@pytest.mark.parametrize("change", ["modified", "untracked"])
def test_dirty_tree_is_refused(tmp_path: Path, change: str) -> None:
    init_repository(tmp_path)
    target = tmp_path / ("README.md" if change == "modified" else "new.py")
    target.write_text("changed\n", encoding="utf-8")

    with pytest.raises(DirtyWorkingTreeError, match="1 uncommitted or untracked"):
        code_revision(tmp_path, allow_dirty=False)


def test_allow_dirty_marks_the_revision(tmp_path: Path) -> None:
    sha = init_repository(tmp_path)
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")

    assert code_revision(tmp_path, allow_dirty=True) == f"{sha}-dirty"


def test_a_directory_outside_git_has_no_revision(tmp_path: Path) -> None:
    with pytest.raises(ProvenanceError, match="rev-parse"):
        code_revision(tmp_path, allow_dirty=True)


def test_dvc_md5_matches_what_dvc_add_records(tmp_path: Path) -> None:
    _write_dvc_fixture(tmp_path)

    assert dvc_md5(tmp_path / "data") == DVC_DIRECTORY_MD5
    assert dvc_md5(tmp_path / "golden.jsonl") == DVC_FILE_MD5


def test_dataset_revision_lists_each_verified_pointer(tmp_path: Path) -> None:
    pointers = _write_dvc_fixture(tmp_path)

    revision = dataset_revision(
        pointers,
        [tmp_path / "data" / "one.jsonl", tmp_path / "golden.jsonl"],
        allow_dirty=False,
    )

    assert revision == (
        f"{pointers[0].as_posix()}={DVC_DIRECTORY_MD5},{pointers[1].as_posix()}={DVC_FILE_MD5}"
    )


def test_data_that_differs_from_its_pointer_is_refused(tmp_path: Path) -> None:
    pointers = _write_dvc_fixture(tmp_path)
    (tmp_path / "data" / "one.jsonl").write_text('{"text": "b"}\n', encoding="utf-8")

    with pytest.raises(DatasetMismatchError, match=r"data\.dvc"):
        dataset_revision(pointers, [], allow_dirty=False)

    revision = dataset_revision(pointers, [], allow_dirty=True)
    assert revision.split(",")[0].endswith(".dir-dirty")
    assert revision.split(",")[1].endswith(DVC_FILE_MD5)


def test_data_outside_the_pointers_is_refused(tmp_path: Path) -> None:
    pointers = _write_dvc_fixture(tmp_path)
    stray = tmp_path / "stray.jsonl"
    stray.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="not covered"):
        dataset_revision(pointers, [stray], allow_dirty=True)


def test_missing_output_and_missing_pointers_are_refused(tmp_path: Path) -> None:
    pointer = write_dvc_pointer(tmp_path / "absent.dvc", "absent", DVC_FILE_MD5)

    with pytest.raises(ProvenanceError, match="dvc pull"):
        dataset_revision([pointer], [], allow_dirty=True)
    with pytest.raises(ProvenanceError, match="No DVC pointers"):
        dataset_revision([], [], allow_dirty=True)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("outs: []\n", "exactly one output"),
        ("outs:\n- md5: abc\n  hash: md5-dos2unix\n  path: x\n", "hash: md5"),
        ("outs:\n- md5: abc\n  hash: md5\n", "no output path"),
    ],
    ids=["no-outputs", "dvc2-hash", "no-path"],
)
def test_malformed_pointers_are_refused(tmp_path: Path, content: str, message: str) -> None:
    pointer = tmp_path / "bad.dvc"
    pointer.write_text(content, encoding="utf-8")

    with pytest.raises(ProvenanceError, match=message):
        read_dvc_pointer(pointer)
