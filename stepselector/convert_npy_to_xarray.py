"""
General converter: NumPy trajectory files (one per observation) -> xarray Dataset -> Zarr.

Key features:
- Accepts .npy of shape (T, F, 2) or (F, 2) per observation.
- Flexible filename parsing via a user regex to extract the observation id.
- Optional observation-level CSV: all kept columns (except 'observation') become Dataset attrs.
- Optional track-level CSV: all kept columns (except 'observation','track') become coords on 'track'.
- Observer UTM is added ONLY if lat/lon exist, look like degrees, and easting/northing are absent.

Chunking tips:
- If you analyze per-track, set chunk_tracks small (1–8).
- For long videos, chunk_frames ~1024–4096 is a good start.

Author: you :)
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import xarray as xr

try:
    import pandas as pd
    HAVE_PANDAS = True
except Exception:
    HAVE_PANDAS = False


# ------------------------- small helpers -------------------------

def _require_pandas():
    if not HAVE_PANDAS:
        raise RuntimeError("pandas is required when supplying metadata CSVs. Please `pip install pandas`.")

def _jsonable_attr(v):
    # Make attrs JSON/Zarr-friendly (plain scalars / strings / None)
    if v is None:
        return None
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    # datetime/timedelta -> string
    try:
        import pandas as pd  # local import
        if isinstance(v, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(v).isoformat()
        if isinstance(v, (pd.Timedelta, np.timedelta64)):
            return str(pd.to_timedelta(v))
        # pandas NA
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (list, tuple)):
        return [_jsonable_attr(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable_attr(val) for k, val in v.items()}
    return v  # str, int, float, bool

def _sanitize_attrs(attrs: dict) -> dict:
    return {str(k): _jsonable_attr(v) for k, v in attrs.items()}

def _looks_like_degrees(lat: float, lon: float) -> bool:
    try:
        lat = float(lat); lon = float(lon)
    except Exception:
        return False
    return (-90.0 <= lat <= 90.0) and (-180.0 <= lon <= 180.0)

def _add_observer_utm_if_needed(attrs: dict) -> dict:
    """
    If attrs contains observer_lat/lon in degrees AND does not already have observer_easting/northing,
    compute UTM and add. Otherwise silently skip.
    """
    if "observer_easting" in attrs or "observer_northing" in attrs:
        return attrs  # already present
    lat = attrs.get("observer_lat", None)
    lon = attrs.get("observer_lon", None)
    if lat is None or lon is None:
        return attrs
    if not _looks_like_degrees(lat, lon):
        return attrs
    try:
        import utm
        e, n, zn, zl = utm.from_latlon(float(lat), float(lon))
        attrs["observer_easting"] = float(e)
        attrs["observer_northing"] = float(n)
        attrs["observer_utm_zone_number"] = int(zn)
        attrs["observer_utm_zone_letter"] = str(zl)
        attrs.setdefault("utm_zone", f"{zn}{zl}")
    except Exception as exc:
        warnings.warn(f"Could not compute observer UTM from lat/lon: {exc}")
    return attrs

def _default_normalize_obs_label(s: str) -> str:
    """
    Normalize observation labels so 'ob123' and 'observation123' become 'observation123'.
    This helps join file-derived names to CSV labels. Override with `normalize_obs_label`.
    """
    s = str(s).strip()
    m = re.fullmatch(r"(?i)observation(\d+)", s)
    if m:
        return f"observation{m.group(1)}"
    m = re.fullmatch(r"(?i)ob(\d+)", s)
    if m:
        return f"observation{m.group(1)}"
    return s  # leave as-is

def _parse_observation_id(fname: str, obs_name_regex: str, normalize) -> str:
    """
    Extract observation id text from filename using the provided regex.
    The regex must have exactly one capturing group OR a named group 'obs'.
    The captured text is normalized via `normalize`.
    """
    name = Path(fname).name
    pat = re.compile(obs_name_regex)
    m = pat.search(name)
    if not m:
        raise ValueError(f"Filename does not match obs_name_regex: {name!r} / {obs_name_regex!r}")
    if "obs" in m.groupdict():
        raw = m.group("obs")
    else:
        if m.lastindex != 1:
            raise ValueError("obs_name_regex must have exactly 1 capturing group (or a named group 'obs').")
        raw = m.group(1)
    return normalize(raw)

def _maybe_filter_columns(df, keep_cols: Optional[Iterable[str]], mandatory: Iterable[str]) -> "pd.DataFrame":
    _require_pandas()
    mandatory = list(mandatory)
    if keep_cols is None:
        # keep everything, but ensure mandatory columns are present
        for c in mandatory:
            if c not in df.columns:
                raise ValueError(f"Metadata is missing required column: {c!r}")
        return df.copy()
    keep = set(keep_cols) | set(mandatory)
    missing = [c for c in mandatory if c not in df.columns]
    if missing:
        raise ValueError(f"Metadata is missing required columns: {missing}")
    present = [c for c in keep if c in df.columns]
    if not present:
        raise ValueError("None of the requested metadata columns are present.")
    return df.loc[:, present].copy()

def _safe_coord_name(name: str) -> str:
    """Avoid conflicts with reserved names."""
    reserved = {"track", "track_index", "frame", "location", "position"}
    n = str(name)
    return n if n not in reserved else f"{n}_meta"


# ------------------------- I/O: metadata -------------------------

def _load_obs_metadata(csv_path: Optional[Path], usecols: Optional[Iterable[str]], normalize) -> Optional["pd.DataFrame"]:
    if csv_path is None:
        return None
    _require_pandas()
    df = pd.read_csv(csv_path)
    if "observation" not in df.columns:
        raise ValueError("Observation metadata CSV must contain a column named 'observation'.")
    df["observation"] = df["observation"].astype(str).map(normalize)
    df = _maybe_filter_columns(df, usecols, mandatory=["observation"])
    return df

def _load_track_metadata(csv_path: Optional[Path], usecols: Optional[Iterable[str]], normalize) -> Optional["pd.DataFrame"]:
    if csv_path is None:
        return None
    _require_pandas()
    df = pd.read_csv(csv_path)
    req = {"observation", "track"}
    if not req.issubset(df.columns):
        raise ValueError("Track metadata CSV must contain columns: 'observation' and 'track'.")
    df["observation"] = df["observation"].astype(str).map(normalize)
    df["track"] = df["track"].astype(int)
    df = _maybe_filter_columns(df, usecols, mandatory=["observation", "track"])
    return df


# ------------------------- core build/write -------------------------

def _build_dataset_for_file(
    npy_path: Path,
    observation_id: str,
    coord_names: list[str],
    float_dtype: str,
    chunk_tracks: Optional[int],
    chunk_frames: Optional[int],
    obs_meta_df: Optional["pd.DataFrame"],
    track_meta_df: Optional["pd.DataFrame"],
) -> xr.Dataset:
    """
    Create an xarray Dataset for a single observation's .npy file.
    """
    arr = np.load(npy_path, allow_pickle=False)
    if arr.ndim == 2 and arr.shape[1] == 2:
        # (F, 2) -> add track axis of length 1
        arr = arr[None, ...]
    if arr.ndim != 3 or arr.shape[2] != 2:
        raise ValueError(f"{npy_path}: expected shape (T,F,2) or (F,2); got {arr.shape}")

    T, F, two = arr.shape
    if two != 2:
        raise ValueError("Last dimension must be size 2 (coordinates).")
    if float_dtype:
        arr = arr.astype(float_dtype, copy=False)

    # Track metadata slice for this observation (optional)
    if track_meta_df is not None:
        tdf = track_meta_df.loc[track_meta_df["observation"] == observation_id].copy()
    else:
        tdf = None

    # Prepare per-track coords
    if tdf is not None and not tdf.empty:
        # keep only valid tracks
        tdf = tdf.loc[tdf["track"].between(0, T - 1)].copy()

        # ensure one row per track [0..T-1] (fill missing with NA)
        have = set(tdf["track"].tolist())
        missing = [i for i in range(T) if i not in have]
        if missing:
            placeholders = {"observation": [observation_id] * len(missing), "track": missing}
            for c in tdf.columns:
                if c not in placeholders:
                    placeholders[c] = [None] * len(missing)
            tdf = pd.concat([tdf, pd.DataFrame(placeholders)], ignore_index=True)

        tdf = tdf.sort_values("track").drop_duplicates(subset=["track"], keep="first").iloc[:T]
        track_index = tdf["track"].to_numpy(np.int32)
        track_labels = [f"{observation_id}_track{int(i):03d}" for i in track_index]
        track_labels = _to_fixed_unicode(track_labels)

        # build a dict of track coords from ALL remaining columns (except observation, track)
        track_coord_vars = {}
        for col in tdf.columns:
            if col in ("observation", "track"):
                continue
            name = _safe_coord_name(col)
            raw_vals = tdf[col].to_numpy()  # object/strings/numbers possible
        
            # ensure length matches T (guard)
            if raw_vals.shape[0] != T:
                tmp = np.empty(T, dtype=object)
                ncopy = min(T, raw_vals.shape[0])
                tmp[:ncopy] = raw_vals[:ncopy]
                if ncopy < T:
                    tmp[ncopy:] = None
                raw_vals = tmp
        
            # If it looks like strings/mixed → fixed unicode; else keep numeric/bool
            if np.issubdtype(np.asarray(raw_vals).dtype, np.number) or np.asarray(raw_vals).dtype == bool:
                vals = np.asarray(raw_vals)  # numeric/bool as-is
            else:
                vals = _to_fixed_unicode(raw_vals)
        
            track_coord_vars[name] = ("track", vals)
    else:
        # fabricate minimal track coords
        track_index = np.arange(T, dtype=np.int32)
        track_labels = [f"{observation_id}_track{int(i):03d}" for i in track_index]
        track_coord_vars = {}

    # Observation-level attrs
    attrs = {
        "observation_id": observation_id,
        "source_file": str(npy_path),
        "coord_names": list(coord_names),
        "units": "UTM meters",
    }
    if obs_meta_df is not None:
        rowdf = obs_meta_df.loc[obs_meta_df["observation"] == observation_id]
        if not rowdf.empty:
            row = rowdf.iloc[0].to_dict()
            row.pop("observation", None)
            attrs.update(row)
    # Sanitize + add observer UTM if needed
    attrs = _sanitize_attrs(attrs)
    attrs = _add_observer_utm_if_needed(attrs)

    # Build Dataset
    coords = {
        "track": ("track", track_labels),
        "track_index": ("track", track_index),
        "frame": ("frame", np.arange(F, dtype=np.int32)),
        "location": ("location", list(coord_names)),
    }
    coords.update(track_coord_vars)

    ds = xr.Dataset(
        data_vars={"position": (("track", "frame", "location"), arr)},
        coords=coords,
        attrs=attrs,
    )

    # Chunking
    chunks = {}
    if chunk_tracks and chunk_tracks > 0:
        chunks["track"] = int(chunk_tracks)
    if chunk_frames and chunk_frames > 0:
        chunks["frame"] = int(chunk_frames)
    if chunks:
        ds = ds.chunk(chunks)

    return ds

def _write_zarr(ds: xr.Dataset, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    obs_id = str(ds.attrs.get("observation_id", "unknown_observation"))
    out_path = out_dir / f"{obs_id}.zarr"
    # compression if available
    encoding = {}
    try:
        import numcodecs
        compressor = numcodecs.Blosc(cname="zstd", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE)
        encoding = {"position": {"compressor": compressor}}
    except Exception:
        encoding = {"position": {}}
    ds.to_zarr(out_path, mode="w", encoding=encoding)
    return out_path


# ------------------------- public API -------------------------

def convert_npy_dir_to_zarr(
    raw_tracks_directory: str | Path,
    output_directory: str | Path,
    obs_name_regex: str,
    coord_names: list[str] = ("easting", "northing"),
    *,
    include_observations: Optional[Iterable[str]] = None,
    normalize_observation_names: bool = False,
    normalize_obs_label = _default_normalize_obs_label,
    float_dtype: str = "float32",
    chunk_tracks: Optional[int] = 1,
    chunk_frames: Optional[int] = 2048,
    observation_metadata_csv: Optional[str | Path] = None,
    observation_usecols: Optional[Iterable[str]] = None,
    track_metadata_csv: Optional[str | Path] = None,
    track_usecols: Optional[Iterable[str]] = None,
) -> list[Path]:
    """
    Convert a directory of .npy files (one per observation) to Zarr Datasets.

    Parameters
    ----------
    raw_tracks_directory : str | Path
        Folder containing .npy files. Each file is one observation.
    output_directory : str | Path
        Where to write <observation_id>.zarr folders.
    obs_name_regex : str
        Regex with ONE capturing group (or named group 'obs') that extracts the observation id from the filename.
        The captured text is normalized via `normalize_obs_label` (default: "ob123" and "observation123" -> "observation123").
    coord_names : list[str]
        Names for the last-axis coordinates (length must be 2). E.g. ["easting","northing"] or ["lon","lat"].
    include_observations : Iterable[str] | None
        Optional subset of observation ids to process (after normalization).
    normalize_obs_label : callable
        Function to normalize observation labels for matching to CSVs.
    float_dtype : str
        dtype for position data, default "float32".
    chunk_tracks : int | None
        Chunk size along track dimension. Set 1–8 for per-track work, or None to leave unchunked.
    chunk_frames : int | None
        Chunk size along frame dimension. For long videos, 1024–4096 is a good start. None to leave unchunked.
    observation_metadata_csv : str | Path | None
        Optional CSV path with observation-level metadata (must include column "observation").
    observation_usecols : Iterable[str] | None
        Optional subset of observation CSV columns to include (plus "observation" which is required).
        If None, all columns are included.
    track_metadata_csv : str | Path | None
        Optional CSV path with track-level metadata (must include "observation" and "track").
    track_usecols : Iterable[str] | None
        Optional subset of track CSV columns to include (plus required cols). If None, all columns are included.

    Returns
    -------
    list[Path]
        List of written Zarr store paths.
    """
    raw_dir = Path(raw_tracks_directory)
    out_dir = Path(output_directory)
    if len(coord_names) != 2:
        raise ValueError("coord_names must have length 2 (e.g. ['easting','northing']).")

    # discover npy files
    npy_files = sorted(raw_dir.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {raw_dir}")

    # prepare metadata (optional)
    if normalize_observation_names:
        obs_meta = _load_obs_metadata(Path(observation_metadata_csv), observation_usecols, normalize_obs_label) if observation_metadata_csv else None
        trk_meta = _load_track_metadata(Path(track_metadata_csv), track_usecols, normalize_obs_label) if track_metadata_csv else None
        normalizer = normalize_obs_label
    else:
        # Load raw CSVs (without normalization)
        obs_meta = _load_obs_metadata(Path(observation_metadata_csv), observation_usecols, lambda x: x) if observation_metadata_csv else None
        trk_meta = _load_track_metadata(Path(track_metadata_csv), track_usecols, lambda x: x) if track_metadata_csv else None
        normalizer = lambda x: x  # identity

    # pick files to process
    selected = []
    for p in npy_files:
        try:
            obs_id = _parse_observation_id(p.name, obs_name_regex, normalizer)
        except Exception as e:
            warnings.warn(f"Skipping file without valid observation id: {p.name} ({e})")
            continue
        if include_observations is not None:
            inc_norm = {normalizer(x) for x in include_observations}
            if obs_id not in inc_norm:
                continue
        selected.append((p, obs_id))

    if not selected:
        raise RuntimeError("No files matched the regex / selection criteria.")

    written = []
    for npy_path, obs_id in selected:
        ds = _build_dataset_for_file(
            npy_path=npy_path,
            observation_id=obs_id,
            coord_names=list(coord_names),
            float_dtype=float_dtype,
            chunk_tracks=chunk_tracks,
            chunk_frames=chunk_frames,
            obs_meta_df=obs_meta,
            track_meta_df=trk_meta,
        )
        out = _write_zarr(ds, out_dir)
        written.append(out)
    return written


# ------------------------- CLI convenience (optional) -------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Convert .npy observation files into xarray/Zarr datasets.")
    ap.add_argument("raw_tracks_directory", type=str, help="Folder containing .npy files.")
    ap.add_argument("output_directory", type=str, help="Output folder for <observation_id>.zarr stores.")
    ap.add_argument("--obs_name_regex", type=str, required=True,
                    help=r"Regex with one capturing group (or named 'obs') extracting the observation id from file name, e.g. '(observation\d+)' or '(ob\d+)'")
    ap.add_argument("--coord_names", nargs=2, default=["easting","northing"], help="Names for coordinate components (length 2).")
    ap.add_argument("--include_observations", nargs="*", default=None, help="Optional list of observation ids to process.")
    ap.add_argument("--float_dtype", type=str, default="float32", help="Float dtype for position data.")
    ap.add_argument("--chunk_tracks", type=int, default=1, help="Chunk size along track dimension (set 0 or omit to disable).")
    ap.add_argument("--chunk_frames", type=int, default=2048, help="Chunk size along frame dimension (set 0 or omit to disable).")
    ap.add_argument("--observation_metadata_csv", type=str, default=None, help="CSV path with observation-level metadata.")
    ap.add_argument("--observation_usecols", nargs="*", default=None, help="Subset of observation metadata columns to include.")
    ap.add_argument("--track_metadata_csv", type=str, default=None, help="CSV path with track-level metadata.")
    ap.add_argument("--track_usecols", nargs="*", default=None, help="Subset of track metadata columns to include.")

    args = ap.parse_args()

    convert_npy_dir_to_zarr(
        raw_tracks_directory=args.raw_tracks_directory,
        output_directory=args.output_directory,
        obs_name_regex=args.obs_name_regex,
        coord_names=args.coord_names,
        include_observations=args.include_observations,
        float_dtype=args.float_dtype,
        chunk_tracks=(args.chunk_tracks or None),
        chunk_frames=(args.chunk_frames or None),
        observation_metadata_csv=args.observation_metadata_csv,
        observation_usecols=args.observation_usecols,
        track_metadata_csv=args.track_metadata_csv,
        track_usecols=args.track_usecols,
    )


def summarize_written(paths: list[Path], mode: str = "names") -> None:
    """
    Pretty-print what was written.
    mode="names": print just the store names (stems).
    mode="paths": print full paths, one per line.
    """
    if not paths:
        print("Wrote 0 stores.")
        return
    if mode == "paths":
        print(f"Wrote {len(paths)} stores:")
        for p in paths:
            print(f"  - {p}")
    else:
        names = ", ".join(Path(p).stem for p in paths)
        parent = Path(paths[0]).parent
        print(f"Wrote {len(paths)} stores to {parent}: {names}")

def _to_fixed_unicode(arr_like) -> np.ndarray:
    """
    Convert an array of python/nullable strings to fixed-width unicode (<U{maxlen}),
    replacing missing (None/NaN/pandas NA) with "".
    """
    import numpy as _np
    try:
        import pandas as _pd
        s = _pd.Series(arr_like, dtype="object")
        s = s.where(~s.isna(), "")         # fill NA with empty string
        # cast to str first to be safe
        vals = s.astype(str).to_numpy()
    except Exception:
        vals = _np.asarray(arr_like, dtype=object)
        # naive missing handling
        vals = _np.array(["" if (v is None or (isinstance(v, float) and _np.isnan(v))) else str(v) for v in vals], dtype=object)
    # compute max length and cast to fixed-width unicode
    maxlen = max((len(x) for x in vals), default=0)
    return vals.astype(f"<U{maxlen}")

