import json
from pathlib import Path
import subprocess
import sys

import pytest

from convest.config import WORKSPACE, load_config, validate_repo_id
from convest.selection import read_episode_list, select_episodes
from convest import pipeline
from convest.verify import verify
from test_pipeline import make_bag


def test_list_bom_comments_commas_and_canonical_ids(tmp_path):
    path = tmp_path / "group.txt"
    path.write_text("\ufeff# 手动分组\n18，episode19, 020\n\n22 # 备注\n23\n")
    parsed = read_episode_list(path)
    assert parsed["requested_episodes"] == ["episode18", "episode19", "episode20", "episode22", "episode23"]


def test_list_inclusive_ranges_mixed_with_single_ids(tmp_path):
    path = tmp_path / "group.txt"
    path.write_text("\ufeff34-39，episode041-episode043, 045-episode046\n48-48 50 # 备注\n")
    assert read_episode_list(path)["requested_episodes"] == [
        f"episode{n}" for n in [34, 35, 36, 37, 38, 39, 41, 42, 43, 45, 46, 48, 50]
    ]


@pytest.mark.parametrize("content,error", [
    ("18\nepisode018", "duplicate"), ("# comment\n", "empty"),
    ("18-20-22", "invalid episode"), ("../episode18", "invalid episode"), ("-18", "invalid episode"),
    ("39-34", "reversed range"), ("34-", "invalid episode"), ("34--39", "invalid episode"),
    ("34-39\n039", "duplicate episode39"), ("34-39,38-42", "duplicate episode38"),
    ("episode034\n34-39", "duplicate episode34"),
])
def test_invalid_list(tmp_path, content, error):
    path = tmp_path / "group.txt"
    path.write_text(content)
    with pytest.raises(ValueError, match=error):
        read_episode_list(path)


def test_selection_is_explicit_and_never_falls_back_to_all(tmp_path):
    path = tmp_path / "group.txt"
    candidates = [{"path": "/bags/episode18", "eligible": True, "reason": None},
                  {"path": "/bags/episode64", "eligible": False, "reason": "incomplete"},
                  {"path": "/bags/episode99", "eligible": True, "reason": None}]
    path.write_text("18\n64")
    with pytest.raises(ValueError, match="episode64"):
        select_episodes(candidates, path)
    selected, report = select_episodes(candidates, path, skip_ineligible=True)
    assert [p["path"] for p in selected] == ["/bags/episode18"]
    assert report["skipped"][0]["episode"] == "episode64"
    path.write_text("18\n123")
    with pytest.raises(ValueError, match="no bag"):
        select_episodes(candidates, path, skip_ineligible=True)
    path.write_text("18")
    with pytest.raises(ValueError, match="ambiguous"):
        select_episodes(candidates + [{**candidates[0], "path": "/other/episode18"}], path)


@pytest.mark.parametrize("value", ["../data", "a/../b", "/a/b", "a/b/c", "name", "a/", "a/b c"])
def test_repo_id_cannot_be_a_filesystem_path(value):
    with pytest.raises(ValueError, match="namespace/dataset"):
        validate_repo_id(value)


def test_append_list_and_empty_output_preserve_old_indices_and_payloads(tmp_path, monkeypatch):
    source, output, listing = tmp_path / "bags", tmp_path / "output", tmp_path / "group.txt"
    for n in (9, 10):
        make_bag(source / f"episode{n}")
    output.mkdir()  # User-created empty directories, including --resume on the first run.
    config = load_config(WORKSPACE / "configs/gello_pi05.yaml")
    config.update(source_root=str(source), output_root=str(output), repo_id="fr3_wuji/test_high")
    monkeypatch.setattr(pipeline, "WORKSPACE", tmp_path)
    listing.write_text("10")
    assert pipeline.convert(config, resume=True, episode_list=listing) == 0
    old_files = {str(p.relative_to(output)): (p.read_bytes(), p.stat().st_mtime_ns)
                 for p in output.rglob("*") if p.suffix in (".parquet", ".mp4")}
    listing.write_text("9-10\n")  # A range containing old and new bags only appends the new episode.
    assert pipeline.convert(config, resume=True, episode_list=listing, workers=2) == 0
    records = pipeline.load_records(output)
    assert [Path(r["source"]).name for r in records] == ["episode10", "episode9"]
    assert [r["episodes"][0]["episode_index"] for r in records] == [0, 1]
    assert verify(output, full_video=True)["episodes"] == 2
    assert json.loads((output / "conversion/summary.json").read_text())["added_bags"] == 1
    assert all((output / path).read_bytes() == content and (output / path).stat().st_mtime_ns == mtime
               for path, (content, mtime) in old_files.items())
    listing.write_text("9")
    assert pipeline.convert(config, resume=True, episode_list=listing) == 0
    summary = json.loads((output / "conversion/summary.json").read_text())
    assert summary["converted_bags"] == 2 and summary["added_bags"] == 0
    assert summary["retained_bags_not_in_current_selection"] == [str(source / "episode10")]
    with pytest.raises(ValueError, match="requires --episode-list"):
        pipeline.convert(config, resume=True)
    with pytest.raises(ValueError, match="differs"):
        pipeline.convert(dict(config, repo_id="fr3_wuji/test_other"), resume=True, episode_list=listing)


def test_ineligible_preflight_does_not_create_output_and_explicit_skip_is_recorded(tmp_path, monkeypatch):
    source, output, listing = tmp_path / "bags", tmp_path / "output", tmp_path / "group.txt"
    for n in (32, 64):
        make_bag(source / f"episode{n}")
    (source / "episode64/collection_state.json").write_text(json.dumps({"state": "incomplete", "finalized": False,
                                                                       "failures": ["gap 210 ms"]}))
    listing.write_text("32\n64")
    config = load_config(WORKSPACE / "configs/gello_pi05.yaml")
    config.update(source_root=str(source), output_root=str(output))
    monkeypatch.setattr(pipeline, "WORKSPACE", tmp_path)
    with pytest.raises(ValueError, match="episode64"):
        pipeline.convert(config, episode_list=listing)
    assert not output.exists()
    assert pipeline.convert(config, episode_list=listing, skip_ineligible=True) == 0
    summary = json.loads((output / "conversion/summary.json").read_text())
    assert summary["converted_bags"] == 1 and summary["skipped_by_collection_state"] == 1
    assert summary["skipped_selected_bags"][0]["episode"] == "episode64"
    assert "gap 210 ms" in summary["skipped_selected_bags"][0]["reason"]


def test_check_list_cli_is_read_only(tmp_path):
    source, listing = tmp_path / "bags", tmp_path / "group.txt"
    make_bag(source / "episode18")
    listing.write_text("18")
    result = subprocess.run([sys.executable, "-m", "convest.cli", "check-list", str(listing),
                             "--source-root", str(source)], capture_output=True, text=True,
                            env={**__import__("os").environ, "PYTHONPATH": str(WORKSPACE / "src")})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["selected_episodes"] == ["episode18"]
    assert not (tmp_path / "output").exists()
