"""Explicit, lazy adapters. Adding a format does not change the commit pipeline."""
from dataclasses import dataclass
from importlib import import_module


SOURCES = {"gello_rosbag2": "convest.sources.gello_rosbag2"}


@dataclass(frozen=True)
class Target:
    recipe: str
    writer: str
    verifier: str
    file_patterns: tuple[str, ...]
    version: str


TARGETS = {
    "lerobot_v21": Target("convest.recipes.pi05", "convest.formats.lerobot_v21",
                          "convest.verify", ("data/chunk-*/episode_*.parquet",
                          "videos/chunk-*/*/episode_*.mp4"), "0.3.0"),
    "act_hdf5": Target("convest.recipes.act", "convest.formats.act_hdf5",
                       "convest.formats.act_hdf5", ("episode_*.hdf5",), "act-1"),
    "dp3_uncropped_zarr": Target("convest.recipes.dp3", "convest.formats.dp3_zarr",
                                  "convest.formats.dp3_zarr", ("dataset_uncropped.zarr",),
                                  "dp3-uncropped-1"),
}


def get_source(name):
    if name not in SOURCES:
        raise ValueError(f"Unknown source_format {name!r}; available: {', '.join(SOURCES)}")
    return import_module(SOURCES[name])


def get_target(name):
    if name not in TARGETS:
        raise ValueError(f"Unknown target_format {name!r}; available: {', '.join(TARGETS)}")
    return TARGETS[name]


def get_writer(name):
    try:
        return import_module(get_target(name).writer)
    except ModuleNotFoundError as exc:
        raise ValueError(f"Missing dependency {exc.name!r} for {name}; run bash scripts/setup.sh") from exc
