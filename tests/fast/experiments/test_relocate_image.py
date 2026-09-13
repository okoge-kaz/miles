import pytest

from experiments.container import relocate_image


def test_relocation_preserves_image_and_waits_for_consumers(tmp_path, monkeypatch):
    source, destination = tmp_path / "old.sqsh", tmp_path / "container/image.sqsh"
    source.write_bytes(b"image bytes")
    destination.parent.mkdir()
    destination.symlink_to(source)
    stat = source.stat()
    spec = dict(
        source=str(source),
        destination=str(destination),
        source_identity={key: getattr(stat, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")},
    )
    monkeypatch.setattr(relocate_image, "active_consumers", lambda _: ["123"])
    with pytest.raises(RuntimeError, match="123"):
        relocate_image.relocate(spec)
    assert source.exists() and destination.is_symlink()
    monkeypatch.setattr(relocate_image, "active_consumers", lambda _: [])
    assert relocate_image.relocate(spec)["status"] == "complete"
    assert not source.exists() and not destination.is_symlink()
    assert destination.read_bytes() == b"image bytes"
    assert destination.stat().st_ino == stat.st_ino


def test_relocation_refuses_to_overwrite_another_image(tmp_path, monkeypatch):
    source, destination = tmp_path / "old.sqsh", tmp_path / "new.sqsh"
    source.write_bytes(b"source")
    destination.write_bytes(b"existing image")
    stat = source.stat()
    spec = dict(
        source=str(source),
        destination=str(destination),
        source_identity={key: getattr(stat, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns")},
    )
    monkeypatch.setattr(relocate_image, "active_consumers", lambda _: [])
    with pytest.raises(FileExistsError):
        relocate_image.relocate(spec)
    assert source.read_bytes() == b"source" and destination.read_bytes() == b"existing image"
