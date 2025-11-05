# zarr_inspect.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Sequence, Union, Optional, Tuple, List, Dict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# Global root folder for *.zarr stores. Set it once via set_zarr_root().
_ZARR_ROOT: Optional[Path] = None

# -------------------------- configuration --------------------------

def set_zarr_root(path: Union[str, Path]) -> None:
    """
    Set the base directory that contains <observation>.zarr stores.
    """
    global _ZARR_ROOT
    _ZARR_ROOT = Path(path)

def _require_root() -> Path:
    if _ZARR_ROOT is None:
        raise RuntimeError("ZARR root is not set. Call set_zarr_root('/path/to/tracks_zarr') first.")
    return _ZARR_ROOT

# -------------------------- utilities --------------------------

def list_observations() -> List[str]:
    """
    Return a sorted list of observation names by scanning the ZARR root (folder.stem).
    """
    root = _require_root()
    return sorted(p.stem for p in root.glob("*.zarr") if p.is_dir())

def open_obs(obs_name: str) -> xr.Dataset:
    """
    Open an observation Dataset from <root>/<obs_name>.zarr
    """
    root = _require_root()
    store = root / f"{obs_name}.zarr"
    if not store.exists():
        raise FileNotFoundError(f"Zarr store not found: {store}")
    return xr.open_zarr(store)

def get_xy_names(ds: xr.Dataset) -> Tuple[str, str]:
    """
    Return the two coordinate names used along `location` (e.g., 'easting','northing').
    """
    locs = ds["location"].values.tolist()
    if len(locs) != 2:
        raise ValueError(f"Expected 2 location components, found {locs}")
    return locs[0], locs[1]

def track_coord_names(ds: xr.Dataset) -> List[str]:
    """
    All coordinate names that vary over the 'track' dimension (i.e., track-level metadata fields).
    Excludes the primary 'track' and 'track_index' coords.
    """
    return [name for name, c in ds.coords.items() if "track" in c.dims and name not in ("track", "track_index")]

def _label_to_index(ds: xr.Dataset, label: str) -> int:
    """
    Map a human-readable track label (the 'track' coord) to its numeric index.
    """
    if "track" not in ds.coords:
        raise KeyError("Dataset has no 'track' coordinate labels.")
    labels = ds["track"].values
    # labels may be fixed-width unicode ndarray
    matches = np.where(labels == label)[0]
    if matches.size == 0:
        raise KeyError(f"Track label not found: {label!r}")
    return int(matches[0])

def resolve_track_indices(ds: xr.Dataset, which: Union[int, str, Sequence[Union[int, str]]]) -> List[int]:
    """
    Accepts a single track (int index or str label), or a sequence of them, and returns
    a list of numeric track indices.
    """
    if isinstance(which, (int, str)):
        which = [which]
    out: List[int] = []
    for w in which:
        if isinstance(w, int):
            if w < 0 or w >= ds.sizes["track"]:
                raise IndexError(f"track index out of range (0..{ds.sizes['track']-1}): {w}")
            out.append(w)
        elif isinstance(w, str):
            out.append(_label_to_index(ds, w))
        else:
            raise TypeError(f"Unsupported track selector type: {type(w)}")
    return out

# -------------------------- user-facing helpers --------------------------

def summarize_observation(obs_name: str) -> None:
    """
    Print summary info for an observation: track/frame counts, common attrs, and available track metadata fields.
    """
    ds = open_obs(obs_name)
    loc0, loc1 = get_xy_names(ds)
    T = ds.sizes["track"]
    F = ds.sizes["frame"]

    print(f"Observation: {obs_name}")
    print(f"  tracks: {T}")
    print(f"  frames: {F}")
    print(f"  location components: {loc0}, {loc1}")

    # Observation-level attributes
    print("\nObservation-level attributes:")
    preferred = [
        "big_map", "site", "date",
        "observer_lat", "observer_lon",
        "observer_easting", "observer_northing",
        "observer_utm_zone_number", "observer_utm_zone_letter",
    ]
    for k in preferred:
        if k in ds.attrs:
            print(f"  - {k}: {ds.attrs[k]}")
    other = [k for k in ds.attrs.keys() if k not in {"observation_id", "source_file", "coord_names", "units", *preferred}]
    for k in sorted(other):
        print(f"  - {k}: {ds.attrs[k]}")

    # Track metadata columns
    cols = track_coord_names(ds)
    print("\nTrack-level metadata columns:")
    if cols:
        print("  " + ", ".join(cols))
    else:
        print("  (none)")

def show_track_metadata(
    obs_name: str,
    track: Union[int, str],
) -> Dict[str, object]:
    """
    Return (and print) per-track metadata for a given track specified by numeric index or label.
    Returns a dict of field -> value for convenience.
    """
    ds = open_obs(obs_name)
    ti = resolve_track_indices(ds, track)[0]
    info: Dict[str, object] = {}

    print(f"{obs_name} — track index {ti:03d}")
    # readable label (if present)
    try:
        label = ds["track"].isel(track=ti).item()
        print(f"  label: {label}")
        info["label"] = label
    except Exception:
        pass

    info["track_index"] = int(ds["track_index"].isel(track=ti))
    print(f"  track_index: {info['track_index']}")

    for name in track_coord_names(ds):
        val = ds[name].isel(track=ti).values
        try:
            if np.ndim(val) == 0:
                val = val.item()
        except Exception:
            pass
        print(f"  {name}: {val}")
        info[name] = val
    return info

def plot_tracks(
    obs_name: str,
    tracks: Union[int, str, Sequence[Union[int, str]]],
    *,
    max_points: Optional[int] = None,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    label_prefix: str = "track",
) -> plt.Axes:
    """
    Plot one or more tracks (by numeric index or label). If max_points is set, frames are downsampled for speed.

    Parameters
    ----------
    obs_name : str
        Observation name (e.g., 'observation036').
    tracks : int | str | sequence of int/str
        Track selectors (numeric indices or label strings).
    max_points : int | None
        If provided and frames > max_points, downsample frames by an integer stride.
    ax : matplotlib Axes | None
        If provided, draw into this axes; otherwise create a new figure.
    show : bool
        Whether to call plt.show() if a new figure is created.
    label_prefix : str
        Prefix used for legend labels ('track' by default).

    Returns
    -------
    matplotlib Axes
    """
    ds = open_obs(obs_name)
    loc0, loc1 = get_xy_names(ds)
    pos = ds["position"]  # (track, frame, location)
    indices = resolve_track_indices(ds, tracks)

    # Optional frame downsampling
    if max_points and ds.sizes["frame"] > max_points:
        step = int(np.ceil(ds.sizes["frame"] / max_points))
        pos = pos.isel(frame=slice(0, ds.sizes["frame"], step))

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
        created_fig = True

    # location order mapping
    locs = ds["location"].values.tolist()
    ix = locs.index(loc0)
    iy = locs.index(loc1)

    for ti in indices:
        xy = pos.isel(track=ti).load().values  # (frame, 2)
        x = xy[:, ix]
        y = xy[:, iy]

        # Use label if available for nicer legend
        try:
            lbl = ds["track"].isel(track=ti).item()
        except Exception:
            lbl = f"{label_prefix} {ti:03d}"

        ax.plot(x, y, label=str(lbl))

    ax.set_xlabel(loc0)
    ax.set_ylabel(loc1)
    ax.set_title(f"{obs_name}: {len(indices)} track(s)")
    ax.axis("equal")
    ax.legend()

    if created_fig and show:
        plt.show()

    return ax
