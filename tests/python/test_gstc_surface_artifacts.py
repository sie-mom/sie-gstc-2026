"""Portable relative paths in copied Windows and Linux run manifests."""

import pytest

from applications.gstc_surface.artifacts import artifact_path


@pytest.mark.parametrize("name", ["case/zx.csv", r"case\zx.csv"])
def test_manifest_paths_resolve_without_rewriting_data(tmp_path, name):
    directory = tmp_path / "case"
    directory.mkdir()
    csv = directory / "zx.csv"
    csv.write_bytes(b"saved field samples\n")
    assert artifact_path(tmp_path, name) == csv
    assert csv.read_bytes() == b"saved field samples\n"


@pytest.mark.parametrize(
    "name",
    [
        "../outside.csv",
        r"..\outside.csv",
        "/outside.csv",
        r"C:\run\zx.csv",
        r"C:zx.csv",
        r"\run\zx.csv",
        r"\\server\share\zx.csv",
    ],
)
def test_foreign_absolute_and_escaping_paths_remain_rejected(tmp_path, name):
    with pytest.raises(ValueError, match="inside its field directory"):
        artifact_path(tmp_path, name)
