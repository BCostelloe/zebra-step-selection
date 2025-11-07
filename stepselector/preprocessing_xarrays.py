from pathlib import Path
from tqdm.notebook import tqdm
import xarray as xr
import warnings
import numpy as np
import glob
import re
import pandas as pd

try:
    import rasterio as rio
    HAVE_RASTERIO = True
except Exception:
    HAVE_RASTERIO = False

try:
    from osgeo import gdal
    HAVE_GDAL = True
except Exception:
    HAVE_GDAL = False

# ----------
# Helper functions
# ----------

def _normalize_obs_ids(obs):
    """Accept 'ob015'/'observation015' or a list thereof; return set of 'observation015' strings."""
    if obs is None:
        return None
    if isinstance(obs, str):
        obs = [obs]
    out = set()
    for s in obs:
        s = s.strip()
        if s.startswith("observation"):
            out.add(s)
        elif s.startswith("ob") and s[2:].isdigit():
            out.add("observation" + s[2:])
        else:
            # if user passed '015', tolerate it
            digits = s.strip("observation").strip("ob")
            if digits.isdigit():
                out.add("observation" + digits)
    return out

def _discover_stores(zarr_root: Path, obs_to_process):
    stores = sorted(zarr_root.glob("observation*.zarr"))
    wanted = _normalize_obs_ids(obs_to_process)
    if wanted is not None:
        stores = [p for p in stores if p.stem in wanted]
        # warn on missing requested obs
        missing = sorted(wanted - {p.stem for p in stores})
        if missing:
            warnings.warn(f"Requested observations not found in {zarr_root}: {missing}")
    if not stores:
        raise FileNotFoundError(f"No matching observation*.zarr in {zarr_root} (after filtering).")
    return stores

def _raster_checker_or_none(zarr_root_store: xr.Dataset, rasters_directory: str | Path | None):
    """
    Return a callable check(x, y)->bool using the observation's DSM if available, else None.
    The DSM path is built from ds.attrs['big_map'] under .../DSMs/{big_map}_dsm.tif.
    """
    if rasters_directory is None:
        return None
    if not HAVE_RASTERIO:
        warnings.warn("rasterio not installed; skipping raster boundary checks.")
        return None

    map_name = zarr_root_store.attrs.get("big_map", None)
    if not map_name:
        warnings.warn("big_map not found in dataset attrs; skipping raster boundary checks.")
        return None

    dsm_path = Path(rasters_directory) / "DSMs" / f"{map_name}_dsm.tif"
    if not dsm_path.exists():
        warnings.warn(f"DSM not found: {dsm_path}. Skipping raster checks for this observation.")
        return None

    src = rio.open(dsm_path)
    nodata = src.nodata if src.nodata is not None else -10000

    def _ok(x: float, y: float) -> bool:
        try:
            # Quick bounds check
            if not (src.bounds.left <= x <= src.bounds.right and src.bounds.bottom <= y <= src.bounds.top):
                return False
            # Sample band 1 at (x,y)
            val = next(src.sample([(x, y)]))[0]
            if np.isnan(val):
                return False
            if nodata is not None and val == nodata:
                return False
            return True
        except Exception:
            return False

    # keep dataset open during this observation; caller should close after
    _ok._rio_src = src
    return _ok

def _walk_equal_steps(x: np.ndarray, y: np.ndarray, frames: np.ndarray,
                      step_length: float, offset: float):
    """
    Given a dense track (x,y,frames) with possible NaNs, return arrays for
    equally spaced steps of length `step_length` (meters) starting `offset` meters
    from the first valid point. Frames are interpolated linearly between neighbors.
    """
    # mask to finite points
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return np.empty(0), np.empty(0), np.empty(0)
    xi = x[finite]; yi = y[finite]; fi = frames[finite]
    n = xi.size
    if n == 0:
        return np.empty(0), np.empty(0), np.empty(0)

    # cumulative distance along polyline
    seg_dx = np.diff(xi)
    seg_dy = np.diff(yi)
    seg_len = np.hypot(seg_dx, seg_dy)
    s = np.concatenate(([0.0], np.cumsum(seg_len)))  # arc-length coordinate

    total_len = s[-1]
    if total_len == 0.0:
        # all points identical -> return the first one if offset==0
        if offset == 0.0:
            return np.array([fi[0]], float), np.array([xi[0]], float), np.array([yi[0]], float)
        else:
            return np.empty(0), np.empty(0), np.empty(0)

    # targets along s: start at offset, then every step_length
    start_s = offset
    if start_s > total_len:
        return np.empty(0), np.empty(0), np.empty(0)
    targets = np.arange(start_s, total_len + 1e-9, step_length, dtype=float)

    # interpolate (piecewise-linear) xi(s), yi(s), fi(s)
    # find, for each t, the segment index k such that s[k] <= t <= s[k+1]
    k = np.searchsorted(s, targets, side="right") - 1
    k = np.clip(k, 0, n - 2)
    # local fraction on the segment
    seg_s = s[k+1] - s[k]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(seg_s > 0, (targets - s[k]) / seg_s, 0.0)

    x_t = xi[k] + frac * (xi[k+1] - xi[k])
    y_t = yi[k] + frac * (yi[k+1] - yi[k])
    f_t = fi[k] + frac * (fi[k+1] - fi[k])

    return f_t.astype(float), x_t.astype(float), y_t.astype(float)

def _group_path(step_length: float, offset: float, structure: str, observed_or_sim: str) -> str:
    """Return zarr group path string according to chosen hierarchy."""
    S = int(step_length)
    O = int(offset)
    if structure == "offset_inside":  # steps_5m/observed/offset_0m
        return f"steps_{S}m/{observed_or_sim}/offset_{O}m"
    elif structure == "observed_inside":  # steps_5m/offset_0m/observed
        return f"steps_{S}m/offset_{O}m/{observed_or_sim}"
    else:
        raise ValueError("structure must be 'offset_inside' or 'observed_inside'")

def _group_observed(L, O): return f"steps_{int(L)}m/offset_{int(O)}m/observed"
def _group_simulated(L, O): return f"steps_{int(L)}m/offset_{int(O)}m/simulated"

def _headings(points_xy: np.ndarray) -> np.ndarray:
    """Absolute step headings (radians) for consecutive pairs."""
    dx = np.diff(points_xy[:, 0])
    dy = np.diff(points_xy[:, 1])
    return np.arctan2(dy, dx)  # [-π, π]

def _step_lengths(points_xy: np.ndarray) -> np.ndarray:
    dx = np.diff(points_xy[:, 0])
    dy = np.diff(points_xy[:, 1])
    return np.hypot(dx, dy)

def _wrap_angle(angle):
    # wrap radians to (-pi, pi]
    return (angle + np.pi) % (2*np.pi) - np.pi

def _deltas_for_observation_from_group(obs_ds: "xr.Dataset") -> np.ndarray:
    """
    Collect turning-angle deltas (in radians, wrapped to (-pi, pi]) for ALL tracks
    in the observed-steps group of a single observation.

    Expects obs_ds to have:
      - position: (point, location) with XY in meters
      - track_start: (track,)
      - track_count: (track,)
    """
    if ("position" not in obs_ds or
        "track_start" not in obs_ds or
        "track_count" not in obs_ds):
        return np.empty(0, dtype=float)

    pos = obs_ds["position"].values          # (point, 2)
    starts = obs_ds["track_start"].values.astype(np.int64)
    counts = obs_ds["track_count"].values.astype(np.int64)

    out = []
    for s, c in zip(starts, counts):
        if c < 3:
            continue  # need at least 3 points -> 2 headings -> 1 delta
        pts = pos[s:s+c, :]
        heads = _headings(pts)                      # length c-1
        deltas = _wrap_angle(heads[1:] - heads[:-1])  # length c-2
        if deltas.size:
            out.append(deltas.astype(float, copy=False))

    return np.concatenate(out) if out else np.empty(0, dtype=float)


def _deltas_for_track_from_group(obs_ds: "xr.Dataset", track_i: int) -> np.ndarray:
    """
    Turning-angle deltas (radians, (-pi, pi]) for a SINGLE track index in the
    observed-steps group.
    """
    if ("position" not in obs_ds or
        "track_start" not in obs_ds or
        "track_count" not in obs_ds):
        return np.empty(0, dtype=float)

    starts = obs_ds["track_start"].values.astype(np.int64)
    counts = obs_ds["track_count"].values.astype(np.int64)
    if track_i < 0 or track_i >= starts.size:
        return np.empty(0, dtype=float)

    s = int(starts[track_i]); c = int(counts[track_i])
    if c < 3:
        return np.empty(0, dtype=float)

    pts = obs_ds["position"].values[s:s+c, :]   # (c,2)
    heads = _headings(pts)                      # (c-1,)
    deltas = _wrap_angle(heads[1:] - heads[:-1])# (c-2,)
    return deltas.astype(float, copy=False)

def _angle_deg_between(vx1, vy1, vx2, vy2):
    dot = vx1*vx2 + vy1*vy2
    n1 = np.hypot(vx1, vy1)
    n2 = np.hypot(vx2, vy2)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = np.clip(dot / (n1*n2), -1.0, 1.0)
    return np.degrees(np.arccos(c))

def _get_observer_xy(root_ds: xr.Dataset):
    ox = root_ds.attrs.get("observer_easting", None)
    oy = root_ds.attrs.get("observer_northing", None)
    if ox is not None and oy is not None:
        return float(ox), float(oy)
    lat = root_ds.attrs.get("observer_lat", None)
    lon = root_ds.attrs.get("observer_lon", None)
    if (lat is not None) and (lon is not None) and HAVE_UTM:
        e, n, _, _ = _utm.from_latlon(float(lat), float(lon))
        return float(e), float(n)
    raise ValueError("Observer coords not found in attrs.")

def _get_xy_arrays(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (x, y) as NP arrays from ds['position'] using the dataset's 'location' labels.
    Assumes position has shape (point, location) and location like ["easting", "northing"].
    """
    if "position" not in ds or "location" not in ds.coords:
        raise ValueError("Dataset must contain 'position' and 'location' (coords).")
    locs = ds["location"].values.tolist()
    try:
        ix_x = locs.index("easting")
        ix_y = locs.index("northing")
    except ValueError:
        # fall back to first/second column
        ix_x, ix_y = 0, 1
    pos = ds["position"].values  # (point, 2)
    return pos[:, ix_x].astype(np.float64), pos[:, ix_y].astype(np.float64)

def _sample_raster_at_xy(src, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Vectorized-ish sampling: returns values for each (x,y).
    Outside-bounds or NoData -> NaN.
    """
    vals = np.full(x.shape[0], np.nan, dtype=np.float32)
    b = src.bounds
    inb = (x >= b.left) & (x <= b.right) & (y >= b.bottom) & (y <= b.top)
    if not inb.any():
        return vals
    coords = list(zip(x[inb], y[inb]))
    # rasterio.sample returns iterable of row arrays per band; we use the first band
    sampled = np.fromiter((v[0] for v in src.sample(coords)), dtype=np.float32, count=inb.sum())
    # handle explicit nodata if set
    nod = src.nodata
    if nod is not None:
        sampled = np.where(sampled == nod, np.nan, sampled)
    vals[inb] = sampled
    return vals

def _xy_from_root(root: xr.Dataset):
    locs = root["location"].values.tolist()
    try:
        ix = locs.index("easting"); iy = locs.index("northing")
    except ValueError:
        ix, iy = 0, 1
    pos = root["position"].values  # (track, frame, 2)
    return pos[..., ix].astype(np.float64), pos[..., iy].astype(np.float64)

def _rowcol(src, x, y):
    r, c = rio.transform.rowcol(src.transform, x, y)
    return r, c

def _los_visible(gdal_band, xA_px, yA_px, zA, xB_px, yB_px, zB) -> bool:
    """Wrapper for GDAL IsLineOfSightVisible; returns bool (False on error)."""
    try:
        res = gdal.IsLineOfSightVisible(
            band=gdal_band,
            xA=int(xA_px), yA=int(yA_px), zA=float(zA),
            xB=int(xB_px), yB=int(yB_px), zB=float(zB),
        )
        return bool(res.is_visible)
    except Exception:
        return False


def _open_group(store: Path, group: str) -> xr.Dataset | None:
    try:
        return xr.open_zarr(store, group=group)
    except Exception:
        return None

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _viewshed_filename(step_id: str, radius_m: float) -> str:
    return f"{step_id}_viewshed{int(radius_m)}m.tif"

def _viewshed_folder(
    viewshed_root: Path,
    observation_id: str,
    track_label: str,         # e.g., "track000"
    is_simulated: bool,
    *,
    dataset_tag: str | None = None,   # e.g., "steps_5m"
    offset: float | None = None       # for simulated only
) -> Path:
    """
    Folder layout (requested):
      - Observed:   <root>/<obs_id>/<trackNNN>/observed/
      - Simulated:  <root>/<obs_id>/<dataset_tag>/<trackNNN>/simulated/offset_<Xm>/
    """
    base = Path(viewshed_root) / observation_id
    if not is_simulated:
        return _ensure_dir(base / track_label / "observed")
    else:
        if dataset_tag is None:
            dataset_tag = "steps_unknown"
        off_tag = f"offset_{_dist_tag(offset or 0.0)}"
        return _ensure_dir(base / dataset_tag / track_label / "simulated" / off_tag)


def _dist_tag(x: float) -> str:
    """Format distances for folder names: 5.0 -> '5m', 2.5 -> '2p5m'."""
    x = float(x)
    return f"{int(x)}m" if x.is_integer() else f"{str(x).replace('.', 'p')}m"


def _viewshed_mean_visible(tif_path: Path) -> float:
    """Read a GTiff and return band mean (0..1). Assumes 1=visible, 0=invisible, -10000 nodata/out-of-range."""
    ds = gdal.Open(str(tif_path))
    try:
        b = ds.GetRasterBand(1)
        # (min,max,mean,std) with approx=0, force=1 to compute
        stats = b.GetStatistics(0, 1)
        if stats is None:
            # fallback: compute array mean ignoring nodata
            arr = b.ReadAsArray()
            nodata = b.GetNoDataValue()
            if nodata is not None:
                m = float(np.nanmean(np.where(arr == nodata, np.nan, arr)))
            else:
                m = float(np.mean(arr))
            return m
        return float(stats[2])  # mean
    finally:
        ds = None

def _generate_viewshed_to_file(
    dsm_path: Path,
    X: float,
    Y: float,
    observer_height: float,
    out_tif: Path,
    radius_m: float,
    threads: int = 1,
):
    """
    Create a viewshed TIFF using GDAL.ViewshedGenerate.
    Encodes 1=visible, 0=invisible, -10000 nodata/out-of-range.
    """
    src_ds = gdal.Open(str(dsm_path))
    if src_ds is None:
        raise FileNotFoundError(f"Cannot open DSM: {dsm_path}")
    try:
        srcBand = src_ds.GetRasterBand(1)
        c_options = [f"NUM_THREADS={int(threads)}", "COMPRESS=PACKBITS"]
        gdal.ViewshedGenerate(
            srcBand=srcBand,
            driverName="GTIFF",
            targetRasterName=str(out_tif),
            creationOptions=c_options,
            observerX=float(X),
            observerY=float(Y),
            observerHeight=float(observer_height),
            targetHeight=0.0,
            visibleVal=1,
            invisibleVal=0,
            outOfRangeVal=-10000,
            noDataVal=-10000,
            dfCurvCoeff=0.85714,
            mode=1,
            maxDistance=float(radius_m),
        )
    finally:
        src_ds = None

def _viewshed_generate_mem(dsm_path: Path, X: float, Y: float, observer_height: float, radius_m: float) -> "gdal.Dataset":
    """
    Create a viewshed in memory (MEM driver) and return the MEM dataset.
    Caller is responsible for closing (by dropping references).
    """
    src_ds = gdal.Open(str(dsm_path))
    if src_ds is None:
        raise FileNotFoundError(f"Cannot open DSM: {dsm_path}")
    try:
        srcBand = src_ds.GetRasterBand(1)
        mem_ds = gdal.ViewshedGenerate(
            srcBand=srcBand,
            driverName="MEM",
            targetRasterName="",      # MEM dataset returned
            creationOptions=[],
            observerX=float(X),
            observerY=float(Y),
            observerHeight=float(observer_height),
            targetHeight=0.0,
            visibleVal=1,
            invisibleVal=0,
            outOfRangeVal=-10000,
            noDataVal=-10000,
            dfCurvCoeff=0.85714,
            mode=1,
            maxDistance=float(radius_m),
        )
        return mem_ds
    finally:
        src_ds = None




def _clean_track_tag(obs_id: str, track_label: str) -> str:
    """
    Make a stable, human-friendly track tag that doesn't duplicate the observation ID.
    Examples:
      obs_id='observation015', track_label='observation015_track000' -> 'track000'
      obs_id='observation015', track_label='track000'                -> 'track000'
      fallback to sanitized version if not string-like.
    """
    s = str(track_label)
    # strip common prefixes that embed the observation id or separators
    for pref in (f"{obs_id}_", f"{obs_id}-"):
        if s.startswith(pref):
            s = s[len(pref):]
            break
    # also handle legacy 'observationNNN_' pattern generically
    if s.startswith("observation") and "_" in s:
        # drop the first token
        s = "_".join(s.split("_")[1:])
    return s or "track_unknown"

def _viewshed_filename_observed(obs_id: str, track_tag: str, frame_int: int, radius_m: float) -> str:
    return f"{obs_id}_{track_tag}_frame{frame_int:06d}_viewshed{int(radius_m)}m.tif"

def _viewshed_filename_sim(obs_id: str, track_tag: str, frame_int: int, rep: int, radius_m: float) -> str:
    return f"{obs_id}_{track_tag}_frame{frame_int:06d}_rep{int(rep)}_viewshed{int(radius_m)}m.tif"

def _viewshed_folder_observed(root: Path, obs_id: str, track_tag: str) -> Path:
    return _ensure_dir(Path(root) / obs_id / track_tag / "observed")

def _viewshed_folder_simulated(root: Path, obs_id: str, track_tag: str, offset: float, dataset_tag: str | None) -> Path:
    base = Path(root) / obs_id
    if dataset_tag:
        base = base / dataset_tag
    return _ensure_dir(base / track_tag / "simulated" / f"offset_{_dist_tag(offset)}")


def _save_mem_to_gtiff(mem_ds: "gdal.Dataset", out_tif: Path, threads: int = 1):
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    gdal.GetDriverByName("GTiff").CreateCopy(
        str(out_tif), mem_ds, options=[f"NUM_THREADS={int(threads)}", "COMPRESS=PACKBITS"]
    )

def _viewshed_mean_from_mem(mem_ds: "gdal.Dataset") -> float:
    band = mem_ds.GetRasterBand(1)
    stats = band.GetStatistics(0, 1)  # (min,max,mean,std)
    return float(stats[2]) if stats is not None else float("nan")

def _viewshed_mean_from_file(tif_path: Path) -> float:
    ds = gdal.Open(str(tif_path))
    try:
        band = ds.GetRasterBand(1)
        stats = band.GetStatistics(0, 1)
        if stats is not None:
            return float(stats[2])
        # fallback: compute with nodata handling
        arr = band.ReadAsArray()
        nodata = band.GetNoDataValue()
        if nodata is not None:
            return float(np.nanmean(np.where(arr == nodata, np.nan, arr)))
        return float(np.mean(arr))
    finally:
        ds = None



def preview_viewshed_io(
    step_length: float,
    offsets: list[float],
    viewshed_save_directory: str | Path,
    *,
    zarr_root: str | Path = "tracks_xarray",
    obs_to_process: str | list[str] | None = None,
    radius: float = 30.0,
    dataset_tag: str | None = None,
    reuse_observed: bool = True,
    reuse_simulated: bool = False,
    limit_per_group: int | None = 200,   # cap rows per (obs,group) in the preview
) -> pd.DataFrame:
    """
    Preview which viewshed files would be reused vs created, using the Step-10
    frame-based naming scheme.

    Returns a pandas DataFrame with columns:
      observation_id, group, is_simulated, offset_m, track_index, track_tag,
      frame, replicate, radius_m, tif_path, exists, action, reason
    """
    zroot = Path(zarr_root)
    outroot = Path(viewshed_save_directory)

    def _auto_tag(L: float) -> str:
        return f"steps_{_dist_tag(L)}"
    sim_tag = dataset_tag or _auto_tag(step_length)

    stores = _discover_stores(zroot, obs_to_process)
    rows = []

    for store in stores:
        root = xr.open_zarr(store)
        obs_id = root.attrs.get("observation_id", store.stem)

        # collect groups present
        group_specs = []
        for off in offsets:
            group_specs.append((_group_observed(step_length, off), False, off))
            group_specs.append((_group_simulated(step_length, off), True,  off))

        for grp, is_sim, off in group_specs:
            ds = _open_group(store, grp)
            if ds is None:
                continue
            needed = ("position", "frame", "observer_height", "track_start", "track_count")
            if not all(k in ds for k in needed):
                continue

            # pull arrays (no heavy compute)
            frames = ds["frame"].values.astype(float)
            frames_int = np.rint(frames).astype(np.int64)

            if is_sim and "replicate" in ds:
                reps = ds["replicate"].values.astype(np.int32)
            else:
                reps = np.zeros(frames_int.shape[0], dtype=np.int32)

            starts = ds["track_start"].values.astype(np.int64)
            counts = ds["track_count"].values.astype(np.int64)
            n_tracks = int(ds.sizes["track"])

            # track tags
            if "track" in ds.coords:
                track_labels = [str(v) for v in ds["track"].values.tolist()]
                track_tags = [_clean_track_tag(obs_id, lab) for lab in track_labels]
            else:
                track_tags = None

            # build preview rows
            made = 0
            for ti in range(n_tracks):
                s = int(starts[ti]); c = int(counts[ti])
                if c <= 0:
                    continue
                track_tag = (track_tags[ti] if track_tags is not None else "track_unknown")
                idxs = np.arange(s, s + c, dtype=np.int64)

                for i in idxs.tolist():
                    frame_i = int(frames_int[i])
                    rep_i = int(reps[i])

                    if is_sim:
                        folder = _viewshed_folder_simulated(outroot, obs_id, track_tag, off, dataset_tag=sim_tag)
                        fname  = _viewshed_filename_sim(obs_id, track_tag, frame_i, rep_i, radius)
                        reuse  = reuse_simulated
                    else:
                        folder = _viewshed_folder_observed(outroot, obs_id, track_tag)
                        fname  = _viewshed_filename_observed(obs_id, track_tag, frame_i, radius)
                        reuse  = reuse_observed

                    tif_path = folder / fname
                    exists = tif_path.exists()
                    action = "reuse" if (reuse and exists) else ("create" if not exists else "ignore")
                    reason = (
                        "reusing existing file"
                        if action == "reuse"
                        else ("will create (missing)" if not exists else "exists but reuse=False")
                    )

                    rows.append({
                        "observation_id": obs_id,
                        "group": grp,
                        "is_simulated": is_sim,
                        "offset_m": float(off),
                        "track_index": ti,
                        "track_tag": track_tag,
                        "frame": frame_i,
                        "replicate": rep_i if is_sim else None,
                        "radius_m": float(radius),
                        "tif_path": str(tif_path),
                        "exists": bool(exists),
                        "action": action,
                        "reason": reason,
                    })

                    made += 1
                    if limit_per_group is not None and made >= int(limit_per_group):
                        break
                if limit_per_group is not None and made >= int(limit_per_group):
                    break

    df = pd.DataFrame(rows, columns=[
        "observation_id","group","is_simulated","offset_m","track_index","track_tag",
        "frame","replicate","radius_m","tif_path","exists","action","reason"
    ])
    return df


def summarize_viewshed_preview(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quick summary table by (is_simulated, action).
    """
    if df.empty:
        return pd.DataFrame({"is_simulated": [], "action": [], "count": []})
    g = df.groupby(["is_simulated","action"]).size().reset_index(name="count")
    g = g.sort_values(["is_simulated","action"]).reset_index(drop=True)
    return g


def _viewshed_mean_visible_from_mem(mem_ds: "gdal.Dataset") -> float:
    band = mem_ds.GetRasterBand(1)
    stats = band.GetStatistics(0, 1)
    if stats is not None:
        return float(stats[2])
    arr = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    if nodata is not None:
        arr = np.where(arr == nodata, np.nan, arr)
    return float(np.nanmean(arr))

def _viewshed_generate_mem_from_band(band: "gdal.Band", X: float, Y: float, observer_height: float, radius_m: float):
    """
    Generate a viewshed *in memory* using an already-open band (supports downsampled MEM DSM).
    """
    return gdal.ViewshedGenerate(
        srcBand=band,
        driverName="MEM",
        targetRasterName="",
        creationOptions=[],
        observerX=float(X),
        observerY=float(Y),
        observerHeight=float(observer_height),
        targetHeight=0.0,
        visibleVal=1,
        invisibleVal=0,
        outOfRangeVal=-10000,
        noDataVal=-10000,
        dfCurvCoeff=0.85714,
        mode=1,
        maxDistance=float(radius_m),
    )

def _save_mem_to_gtiff(mem_ds: "gdal.Dataset", out_tif: Path, threads: int = 1):
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    drv = gdal.GetDriverByName("GTiff")
    drv.CreateCopy(str(out_tif), mem_ds, options=[f"NUM_THREADS={int(threads)}", "COMPRESS=PACKBITS"])

def _build_downsampled_dsm_mem(src_path: Path, target_cell_size_m: float, resample_alg: str = "bilinear") -> "gdal.Dataset":
    """
    Downsample the DSM once per observation into a MEM dataset at target_cell_size_m.
    If native resolution ~ target (within ~5%), returns the original file dataset.
    Caller should keep the returned handle alive for the whole observation.
    """
    src = gdal.Open(str(src_path))
    if src is None:
        raise FileNotFoundError(f"Cannot open DSM: {src_path}")

    gt = src.GetGeoTransform()
    px = abs(gt[1]); py = abs(gt[5]) if gt[5] != 0 else px

    # If within ~5%, reuse original
    if (abs(px - target_cell_size_m)/px < 0.05) and (abs(py - target_cell_size_m)/py < 0.05):
        return src  # keep original open

    # Build WarpOptions with MEM format (this is the key change)
    opts = gdal.WarpOptions(
        format="MEM",
        xRes=target_cell_size_m,
        yRes=target_cell_size_m,
        resampleAlg=resample_alg,   # e.g., "bilinear", "nearest", "average"
        multithread=True,
    )

    # Create an in-memory (MEM) warped dataset
    warped = gdal.Warp(
        destNameOrDestDS="",        # empty string is fine when format='MEM' is set in options
        srcDSOrSrcDSTab=src,
        options=opts
    )

    # Release original; keep warped
    src = None
    if warped is None:
        raise RuntimeError(
            f"GDAL Warp failed for {src_path} at {target_cell_size_m} m (resample={resample_alg}). "
            "Ensure the MEM driver is available in your GDAL build."
        )
    return warped


# ----------
# --- Step 1: Densely interpolate trajectories & write ragged 'dense' group per observation ---
# ----------
        
def densely_interpolate(x: np.ndarray, y: np.ndarray, tolerance: float):
    """
    Insert points so no segment exceeds `tolerance` distance (meters).
    NaN->NaN segments are represented with a single NaN row (and a trailing NaN at the last segment end).
    Returns (new_frames, new_x, new_y) as float64.
    """
    n = len(x)
    if n < 2:
        frames = np.arange(n, dtype=np.float64)
        return frames, x.astype(np.float64, copy=False), y.astype(np.float64, copy=False)

    counts = np.zeros(n - 1, dtype=np.int64)
    total_points = 0

    for i in range(n - 1):
        xi, xj = x[i], x[i + 1]
        yi, yj = y[i], y[i + 1]
        if np.isnan(xi) or np.isnan(xj):
            total_points += 1
            if i + 1 == n - 1:
                total_points += 1
        else:
            seg_len = float(np.hypot(xj - xi, yj - yi))
            num = max(int(seg_len / tolerance), 2)
            counts[i] = num
            total_points += num

    new_x = np.empty(total_points, dtype=np.float64)
    new_y = np.empty(total_points, dtype=np.float64)
    new_f = np.empty(total_points, dtype=np.float64)

    idx = 0
    for i in range(n - 1):
        num = counts[i]
        if num == 0:
            new_x[idx] = np.nan
            new_y[idx] = np.nan
            new_f[idx] = float(i)
            if i + 1 == n - 1:
                new_x[idx + 1] = np.nan
                new_y[idx + 1] = np.nan
                new_f[idx + 1] = float(i + 1)
            idx += 1
        else:
            end = idx + num
            new_x[idx:end] = np.linspace(x[i], x[i + 1], num)
            new_y[idx:end] = np.linspace(y[i], y[i + 1], num)
            new_f[idx:end] = np.linspace(i, i + 1, num, dtype=np.float64)
            idx = end

    return new_f, new_x, new_y


def write_dense_to_zarr_groups(
    zarr_root: str | Path = "tracks_xarray",
    tolerance: float = 0.05,
    obs_to_process: str | list[str] | None = None,
    group_name: str = "dense",
    dtype: str = "float32",
    chunks_point: int | None = 200_000,
):
    """
    Process all observations in `zarr_root` or only those listed in `obs_to_process`.

    Root layout expected (matches your store):
      position(track, frame, location), location(location), track(track), frame(frame),
      and per-track coords: individual_ID, species, age, track_index.

    Writes subgroup `<group_name>` containing ragged dense arrays.
    """
    zarr_root = Path(zarr_root)
    stores = _discover_stores(zarr_root, obs_to_process)

    for store in tqdm(stores, desc="STEP 1: dense interpolation"):
        ds = xr.open_zarr(store)

        # validate expected names from your file
        if "position" not in ds or "track" not in ds.dims or "frame" not in ds.dims or "location" not in ds.coords:
            warnings.warn(f"{store}: unexpected layout; expected position(track,frame,location). Skipping.")
            continue

        loc_labels = ds["location"].compute().values.tolist()
        if len(loc_labels) != 2:
            raise ValueError(f"{store}: location must have length 2, got {len(loc_labels)}")

        x_da = ds["position"].sel(location=loc_labels[0])
        y_da = ds["position"].sel(location=loc_labels[1])
        n_tracks = ds.sizes["track"]

        frames_list, xs, ys = [], [], []
        counts = np.empty(n_tracks, dtype=np.int64)

        for ti in range(n_tracks):
            x = x_da.isel(track=ti).compute().values
            y = y_da.isel(track=ti).compute().values
            f_new, x_new, y_new = densely_interpolate(x, y, tolerance)
            frames_list.append(f_new); xs.append(x_new); ys.append(y_new)
            counts[ti] = len(x_new)

        starts = np.empty_like(counts)
        np.cumsum(np.concatenate([[0], counts[:-1]]), out=starts)

        total_points = int(starts[-1] + counts[-1]) if n_tracks > 0 else 0
        if total_points == 0:
            warnings.warn(f"{store}: no points after interpolation; skipping write.")
            continue

        frame_point = np.concatenate(frames_list).astype(np.float64, copy=False)
        x_point = np.concatenate(xs).astype(dtype, copy=False)
        y_point = np.concatenate(ys).astype(dtype, copy=False)
        pos_point = np.stack([x_point, y_point], axis=1)

        # carry per-track labels/coords
        coords = {
            "track": (("track",), ds["track"].compute().values),
            "location": (("location",), np.array(loc_labels, dtype=object)),
        }
        for name in ("individual_ID", "species", "age", "track_index"):
            if name in ds.coords and "track" in ds[name].dims:
                coords[name] = (("track",), ds[name].compute().values)

        dense = xr.Dataset(
            data_vars={
                "position": (("point", "location"), pos_point),
                "frame":    (("point",), frame_point),
                "track_start": (("track",), starts),
                "track_count": (("track",), counts),
            },
            coords=coords,
            attrs={
                "source_group": "/",
                "description": f"Densely interpolated tracks; tolerance={tolerance} m (ragged contiguous).",
                "units_position": "meters",
                "frame_is_fractional": True,
            },
        )

        dense["position"] = dense["position"].astype(dtype)
        if chunks_point:
            dense = dense.chunk({"point": chunks_point, "location": 2})

        dense.to_zarr(store, mode="w", group=group_name)

# ----------
# --- Step 2: Discretize trajectories & write reference steps ---
# ----------

def extract_observed_steps_to_zarr(
    step_length: float,
    offsets: list[float],
    zarr_root: str | Path = "tracks_xarray",
    dense_group: str = "dense",
    rasters_directory: str | Path | None = None,
    obs_to_process: str | list[str] | None = None,
    dtype: str = "float32",
    chunks_point: int | None = 200_000,
    show_progress: bool = True,
    group_structure: str = "observed_inside",  # "offset_inside" (default) or "observed_inside"
):
    """
    Discretize dense trajectories into equal-length steps for each offset
    and write ragged datasets under each observation's zarr store.

    group_structure:
      - "offset_inside":   steps_{L}m/observed/offset_{O}m     (default)
      - "observed_inside": steps_{L}m/offset_{O}m/observed
    """
    zarr_root = Path(zarr_root)
    stores = _discover_stores(zarr_root, obs_to_process)

    obs_iter = tqdm(stores, desc="Step 2: observations", unit="obs") if show_progress else stores
    for store in obs_iter:
        root_ds = xr.open_zarr(store)
        dense = xr.open_zarr(store, group=dense_group)

        for needed in ("position", "frame", "track_start", "track_count"):
            if needed not in dense:
                raise ValueError(f"{store}/{dense_group}: missing '{needed}'")

        if "location" not in dense.coords or "track" not in dense.coords:
            raise ValueError(f"{store}/{dense_group}: expected 'location' and 'track' coords")

        loc_labels = dense["location"].compute().values.tolist()
        if len(loc_labels) != 2:
            raise ValueError(f"{store}: 'location' must have length 2, got {len(loc_labels)}")
        x_name, y_name = loc_labels[0], loc_labels[1]

        checker = _raster_checker_or_none(root_ds, rasters_directory)

        n_tracks = dense.sizes["track"]
        starts = dense["track_start"].compute().values.astype(np.int64)
        counts = dense["track_count"].compute().values.astype(np.int64)
        pos = dense["position"].compute().values  # (point, 2)
        fr  = dense["frame"].compute().values    # (point,)

        carry_coords = {"track": dense["track"].compute().values,
                        "location": np.array([x_name, y_name], dtype=object)}
        for name in ("individual_ID", "species", "age", "track_index"):
            if name in dense.coords:
                carry_coords[name] = dense[name].compute().values

        offset_iter = tqdm(offsets, desc=f"{store.stem}: offsets", unit="off", leave=False) if show_progress else offsets
        for offset in offset_iter:
            out_frames = []
            out_x = []
            out_y = []
            out_counts = np.empty(n_tracks, dtype=np.int64)

            track_iter = range(n_tracks)
            if show_progress:
                track_iter = tqdm(track_iter, desc=f"{store.stem}: tracks@{offset}m", unit="trk", leave=False)

            for ti in track_iter:
                s = starts[ti]; c = counts[ti]
                if c <= 1:
                    out_counts[ti] = 0
                    continue
                x = pos[s:s+c, 0]
                y = pos[s:s+c, 1]
                f = fr[s:s+c]

                f_steps, x_steps, y_steps = _walk_equal_steps(x, y, f, step_length=step_length, offset=offset)

                if checker is not None and f_steps.size:
                    keep = np.fromiter((checker(xi, yi) for xi, yi in zip(x_steps, y_steps)),
                                       count=f_steps.size, dtype=bool)
                    f_steps = f_steps[keep]; x_steps = x_steps[keep]; y_steps = y_steps[keep]

                out_frames.append(f_steps)
                out_x.append(x_steps)
                out_y.append(y_steps)
                out_counts[ti] = int(x_steps.size)

            out_starts = np.empty_like(out_counts)
            np.cumsum(np.concatenate([[0], out_counts[:-1]]), out=out_starts)
            total_points = int(out_starts[-1] + out_counts[-1]) if n_tracks > 0 else 0

            group_path = _group_path(step_length, offset, group_structure, "observed")

            if total_points == 0:
                # write/overwrite a minimal empty dataset so downstream code can still open the group
                ds_out = xr.Dataset(
                    coords={"track": (("track",), carry_coords["track"]),
                            "location": (("location",), carry_coords["location"])},
                    attrs={"description": f"Observed steps (empty): L={step_length}m, offset={offset}m",
                           "step_length_m": float(step_length),
                           "offset_m": float(offset),
                           "source_group": f"/{dense_group}"},
                )
                ds_out.to_zarr(store, mode="w", group=group_path)
                continue

            frames_point = np.concatenate(out_frames).astype(np.float64, copy=False)
            x_point = np.concatenate(out_x).astype(dtype, copy=False)
            y_point = np.concatenate(out_y).astype(dtype, copy=False)
            pos_point = np.stack([x_point, y_point], axis=1)

            ds_out = xr.Dataset(
                data_vars={
                    "position": (("point", "location"), pos_point),
                    "frame":    (("point",), frames_point),
                    "track_start": (("track",), out_starts),
                    "track_count": (("track",), out_counts),
                },
                coords={
                    "track": (("track",), carry_coords["track"]),
                    "location": (("location",), carry_coords["location"]),
                    **{k: (("track",), v) for k, v in carry_coords.items() if k not in ("track", "location")}
                },
                attrs={
                    "description": "Observed, discretized steps (ragged contiguous).",
                    "step_length_m": float(step_length),
                    "offset_m": float(offset),
                    "source_group": f"/{dense_group}",
                    "units_position": "meters",
                    "frame_is_fractional": True,
                },
            )

            if chunks_point:
                ds_out = ds_out.chunk({"point": chunks_point, "location": 2})

            ds_out.to_zarr(store, mode="w", group=group_path)

        if checker is not None and hasattr(checker, "_rio_src"):
            try:
                checker._rio_src.close()
            except Exception:
                pass

# ----------
# --- Step 3: Simulate reference steps ---
# ----------

def simulate_reference_steps_to_zarr(
    n_sim_per_step: int,
    step_length: float,
    offsets: list[float],
    zarr_root: str | Path = "tracks_xarray",
    rasters_directory: str | Path | None = None,
    obs_to_process: str | list[str] | None = None,
    angle_dist: str = "uniform",  # "uniform" or "observed"
    observed_angle_pool: str = "track",  # "track" | "observation" | "global"
    pool_offsets: list[float] | None = None,  # which offsets to include in the pool (defaults to `offsets`)
    pool_obs_to_process: str | list[str] | None = None,  # only used if observed_angle_pool="global" (defaults to all stores)
    dtype: str = "float32",
    chunks_point: int | None = 400_000,
    show_progress: bool = True,
):
    """
    Generate `n_sim_per_step` simulated steps per observed step and write to:
      steps_{L}m/offset_{O}m/simulated

    When angle_dist="observed", draw Δ-angles from:
      - "track":       same track (pooled across `pool_offsets`)
      - "observation": all tracks in same observation (pooled across `pool_offsets`)
      - "global":      all observations (pooled across `pool_offsets`, and across stores from `pool_obs_to_process` or all)
    """
    zroot = Path(zarr_root)
    stores = _discover_stores(zroot, obs_to_process)

    # Precompute GLOBAL pool if requested
    global_deltas = None
    if angle_dist == "observed" and observed_angle_pool == "global":
        pool_offsets_eff = pool_offsets if pool_offsets is not None else offsets
        pool_stores = _discover_stores(zroot, pool_obs_to_process)  # all by default
        glob = []
        it = tqdm(pool_stores, desc="Step 3: building global Δ pool", unit="obs") if show_progress else pool_stores
        for st in it:
            for off in pool_offsets_eff:
                grp = _group_observed(step_length, off)
                try:
                    obs_ds = xr.open_zarr(st, group=grp)
                except Exception:
                    continue
                deltas_obs = _deltas_for_observation_from_group(obs_ds)
                if deltas_obs.size:
                    glob.append(deltas_obs)
        global_deltas = np.concatenate(glob) if glob else np.empty(0)

    obs_iter = tqdm(stores, desc="Step 3: observations", unit="obs") if show_progress else stores
    for store in obs_iter:
        root_ds = xr.open_zarr(store)
        checker = _raster_checker_or_none(root_ds, rasters_directory)

        off_iter = tqdm(offsets, desc=f"{store.stem}: offsets", leave=False) if show_progress else offsets
        for offset in off_iter:
            grp_obs = _group_observed(step_length, offset)
            try:
                obs = xr.open_zarr(store, group=grp_obs)
            except Exception as e:
                warnings.warn(f"{store}: could not open observed group '{grp_obs}': {e}")
                continue

            # minimal checks
            for needed in ("position", "frame", "track_start", "track_count"):
                if needed not in obs:
                    warnings.warn(f"{store}/{grp_obs}: missing '{needed}', skipping.")
                    continue
            if "location" not in obs.coords or "track" not in obs.coords:
                warnings.warn(f"{store}/{grp_obs}: expected 'location' and 'track' coords, skipping.")
                continue

            loc_labels = obs["location"].compute().values.tolist()
            if len(loc_labels) != 2:
                warnings.warn(f"{store}: 'location' must have length 2; got {len(loc_labels)}")
                continue

            pos = obs["position"].compute().values
            frames = obs["frame"].compute().values
            starts = obs["track_start"].compute().values.astype(np.int64)
            counts = obs["track_count"].compute().values.astype(np.int64)
            n_tracks = obs.sizes["track"]

            carry = {"track": obs["track"].compute().values,
                     "location": np.array(loc_labels, dtype=object)}
            for name in ("individual_ID", "species", "age", "track_index"):
                if name in obs.coords:
                    carry[name] = obs[name].compute().values

            # Build OBSERVATION-level pool (if requested), pooled across pool_offsets
            obs_level_deltas = None
            if angle_dist == "observed" and observed_angle_pool == "observation":
                pool_offsets_eff = pool_offsets if pool_offsets is not None else offsets
                out = []
                for off_pool in pool_offsets_eff:
                    grp_pool = _group_observed(step_length, off_pool)
                    try:
                        obs_pool = xr.open_zarr(store, group=grp_pool)
                    except Exception:
                        continue
                    d = _deltas_for_observation_from_group(obs_pool)
                    if d.size: out.append(d)
                obs_level_deltas = np.concatenate(out) if out else np.empty(0)

            sim_frames_all = []
            sim_x_all = []
            sim_y_all = []
            sim_step_index_all = []
            sim_repl_all = []
            counts_sim = np.empty(n_tracks, dtype=np.int64)

            tr_iter = range(n_tracks)
            if show_progress:
                tr_iter = tqdm(tr_iter, desc=f"{store.stem}: tracks@{offset}m", unit="trk", leave=False)

            for ti in tr_iter:
                s = starts[ti]; c = counts[ti]
                if c < 2:
                    counts_sim[ti] = 0
                    continue

                pts = pos[s:s+c, :]
                frs = frames[s:s+c]
                heads = _headings(pts)       # length c-1
                dists = _step_lengths(pts)   # length c-1

                # TRACK-level pool (if requested), pooled across pool_offsets
                track_level_deltas = None
                if angle_dist == "observed" and observed_angle_pool == "track":
                    pool_offsets_eff = pool_offsets if pool_offsets is not None else offsets
                    out = []
                    for off_pool in pool_offsets_eff:
                        grp_pool = _group_observed(step_length, off_pool)
                        try:
                            obs_pool = xr.open_zarr(store, group=grp_pool)
                        except Exception:
                            continue
                        d = _deltas_for_track_from_group(obs_pool, ti)
                        if d.size: out.append(d)
                    track_level_deltas = np.concatenate(out) if out else np.empty(0)

                def sample_abs_heading_uniform(m):
                    return np.random.uniform(0.0, 2*np.pi, size=m)

                def sample_delta(m):
                    if observed_angle_pool == "track":
                        src = track_level_deltas
                    elif observed_angle_pool == "observation":
                        src = obs_level_deltas
                    else:  # global
                        src = global_deltas
                    if src is not None and src.size:
                        idx = np.random.randint(0, src.size, size=m)
                        return src[idx]
                    # fallback if pool is empty
                    return np.random.uniform(-np.pi, np.pi, size=m)

                # Simulate per observed step
                sim_frames = []
                sim_x = []
                sim_y = []
                sim_step_index = []
                sim_repl = []

                for j in range(c - 1):
                    start_x, start_y = pts[j, 0], pts[j, 1]
                    step_len = float(dists[j])
                    target_frame = float(frs[j+1])

                    got = 0
                    guard = 0
                    max_trials = n_sim_per_step * 50

                    while got < n_sim_per_step and guard < max_trials:
                        guard += 1
                        if angle_dist == "uniform":
                            theta = sample_abs_heading_uniform(1)[0]
                        else:
                            # base = incoming heading at start of this step
                            base = heads[j-1] if j > 0 else heads[0]
                            delta = sample_delta(1)[0]
                            theta = _wrap_angle(base + delta)

                        new_x = start_x + step_len * np.cos(theta)
                        new_y = start_y + step_len * np.sin(theta)
                        if checker is not None and not checker(new_x, new_y):
                            continue

                        sim_x.append(new_x); sim_y.append(new_y)
                        sim_frames.append(target_frame)
                        sim_step_index.append(j)
                        sim_repl.append(got)
                        got += 1

                if sim_x:
                    sim_x_all.append(np.asarray(sim_x, dtype=dtype))
                    sim_y_all.append(np.asarray(sim_y, dtype=dtype))
                    sim_frames_all.append(np.asarray(sim_frames, dtype=float))
                    sim_step_index_all.append(np.asarray(sim_step_index, dtype=np.int32))
                    sim_repl_all.append(np.asarray(sim_repl, dtype=np.int16))
                    counts_sim[ti] = len(sim_x)
                else:
                    counts_sim[ti] = 0

            # pack & write
            starts_sim = np.empty_like(counts_sim)
            np.cumsum(np.concatenate([[0], counts_sim[:-1]]), out=starts_sim)
            total_points = int(starts_sim[-1] + counts_sim[-1]) if n_tracks > 0 else 0

            grp_sim = _group_simulated(step_length, offset)
            if total_points == 0:
                ds_out = xr.Dataset(
                    coords={"track": (("track",), carry["track"]),
                            "location": (("location",), carry["location"])},
                    attrs={"description": f"Simulated steps (empty): L={step_length}m, offset={offset}m",
                           "n_sim_per_step": int(n_sim_per_step),
                           "angle_dist": str(angle_dist),
                           "observed_angle_pool": str(observed_angle_pool),
                           "paired_on": "track + target_step_index + frame"},
                )
                ds_out.to_zarr(store, mode="w", group=grp_sim)
                if checker is not None and hasattr(checker, "_rio_src"):
                    try: checker._rio_src.close()
                    except Exception: pass
                continue

            frames_point = np.concatenate(sim_frames_all).astype(np.float64, copy=False)
            x_point = np.concatenate(sim_x_all).astype(dtype, copy=False)
            y_point = np.concatenate(sim_y_all).astype(dtype, copy=False)
            pos_point = np.stack([x_point, y_point], axis=1)
            step_index_point = np.concatenate(sim_step_index_all).astype(np.int32, copy=False)
            repl_point = np.concatenate(sim_repl_all).astype(np.int16, copy=False)

            ds_out = xr.Dataset(
                data_vars={
                    "position": (("point", "location"), pos_point),
                    "frame":    (("point",), frames_point),
                    "track_start": (("track",), starts_sim),
                    "track_count": (("track",), counts_sim),
                    "target_step_index": (("point",), step_index_point),
                    "replicate": (("point",), repl_point),
                },
                coords={
                    "track": (("track",), carry["track"]),
                    "location": (("location",), carry["location"]),
                    **{k: (("track",), v) for k, v in carry.items() if k not in ("track", "location")}
                },
                attrs={
                    "description": "Simulated reference steps (ragged contiguous).",
                    "step_length_m": float(step_length),
                    "offset_m": float(offset),
                    "n_sim_per_step": int(n_sim_per_step),
                    "angle_dist": str(angle_dist),
                    "observed_angle_pool": str(observed_angle_pool),
                    "paired_on": "track + target_step_index + frame",
                    "source_group_observed": grp_obs,
                    "units_position": "meters",
                },
            )

            if chunks_point:
                ds_out = ds_out.chunk({"point": chunks_point, "location": 2})

            ds_out.to_zarr(store, mode="w", group=grp_sim)

        if checker is not None and hasattr(checker, "_rio_src"):
            try: checker._rio_src.close()
            except Exception: pass


# ----------
# --- Step 4: Get step features relative to observation team ---
# ----------

def annotate_steps_with_observer_features(
    step_length: float,
    offsets: list[float],
    zarr_root: str | Path = "tracks_xarray",
    obs_to_process: str | list[str] | None = None,
    fps: float = 30.0,
    dtype: str = "float32",
    chunks_point: int | None = None,
    show_progress: bool = True,
):
    zroot = Path(zarr_root)
    stores = _discover_stores(zroot, obs_to_process)

    def _set_var_attrs(ds: xr.Dataset, is_sim: bool):
        # add units + descriptions
        desc_obs_sim = "simulated step endpoint" if is_sim else "observed step endpoint"
        attrs_map = {
            "angle_to_observer": {"units": "degree", "description": f"Unsigned angle between step vector and vector from start to observer ({desc_obs_sim})."},
            "dist_to_observer": {"units": "m", "description": f"Distance from {desc_obs_sim} to observer."},
            "delta_observer_dist": {"units": "m", "description": "Change in distance to observer: end - start."},
            "step_length_m": {"units": "m", "description": "Euclidean step length."},
            "step_duration_s": {"units": "s", "description": f"Step duration based on frame difference at {fps} fps."},
            "step_speed_mps": {"units": "m s-1", "description": "Step length divided by duration."},
            "heading_deg": {"units": "degree", "description": "Absolute heading of the step in (-180, 180]."},
        }
        for v, a in attrs_map.items():
            if v in ds:
                ds[v].attrs.update(a)

    obs_iter = tqdm(stores, desc="Step 4: observations", unit="obs") if show_progress else stores
    for store in obs_iter:
        root_ds = xr.open_zarr(store)
        ox, oy = _get_observer_xy(root_ds)

        off_iter = tqdm(offsets, desc=f"{store.stem}: offsets", leave=False) if show_progress else offsets
        for offset in off_iter:
            # ---------- OBSERVED ----------
            grp_obs = _group_observed(step_length, offset)
            try:
                obs = xr.open_zarr(store, group=grp_obs)
            except Exception:
                obs = None

            if obs is not None:
                for needed in ("position", "frame", "track_start", "track_count"):
                    if needed not in obs:
                        warnings.warn(f"{store}/{grp_obs}: missing '{needed}', skipping observed features.")
                        obs = None
                        break

            if obs is not None:
                pos = obs["position"].values           # (point, 2)
                fr  = obs["frame"].values
                starts = obs["track_start"].values.astype(np.int64)
                counts = obs["track_count"].values.astype(np.int64)
                n_tracks = obs.sizes["track"]

                npt = pos.shape[0]
                feat_angle = np.full(npt, np.nan, np.float32)
                feat_dist_end = np.full(npt, np.nan, np.float32)
                feat_delta = np.full(npt, np.nan, np.float32)
                feat_len = np.full(npt, np.nan, np.float32)
                feat_dur = np.full(npt, np.nan, np.float32)
                feat_spd = np.full(npt, np.nan, np.float32)
                feat_head = np.full(npt, np.nan, np.float32)  # <- heading_deg

                for ti in range(n_tracks):
                    s = starts[ti]; c = counts[ti]
                    if c < 2: continue
                    pts = pos[s:s+c, :]
                    frs = fr[s:s+c]

                    dx = np.diff(pts[:, 0]); dy = np.diff(pts[:, 1])
                    step_len = np.hypot(dx, dy).astype(np.float32)
                    step_dur = ((frs[1:] - frs[:-1]) / float(fps)).astype(np.float32)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        step_spd = (step_len / step_dur).astype(np.float32)

                    dist_end = np.hypot(pts[1:, 0] - ox, pts[1:, 1] - oy).astype(np.float32)
                    dist_start = np.hypot(pts[:-1, 0] - ox, pts[:-1, 1] - oy).astype(np.float32)
                    delta = (dist_end - dist_start).astype(np.float32)

                    ang = _angle_deg_between(dx, dy, (ox - pts[:-1, 0]), (oy - pts[:-1, 1])).astype(np.float32)

                    # heading in degrees (-180, 180] for each step
                    heading_deg = (np.degrees(np.arctan2(dy, dx))).astype(np.float32)

                    idx = np.arange(s + 1, s + c, dtype=np.int64)
                    feat_angle[idx] = ang
                    feat_dist_end[idx] = dist_end
                    feat_delta[idx] = delta
                    feat_len[idx] = step_len
                    feat_dur[idx] = step_dur
                    feat_spd[idx] = step_spd
                    feat_head[idx] = heading_deg

                ds_obs_feats = xr.Dataset(
                    data_vars={
                        "angle_to_observer": (("point",), feat_angle),
                        "dist_to_observer": (("point",), feat_dist_end),
                        "delta_observer_dist": (("point",), feat_delta),
                        "step_length_m": (("point",), feat_len),
                        "step_duration_s": (("point",), feat_dur),
                        "step_speed_mps": (("point",), feat_spd),
                        "heading_deg": (("point",), feat_head),
                    },
                    coords={"point": np.arange(npt)},
                )

                if chunks_point:
                    ds_obs_feats = ds_obs_feats.chunk({"point": chunks_point})

                _set_var_attrs(ds_obs_feats, is_sim=False)
                ds_obs_feats.to_zarr(store, mode="a", group=grp_obs)

            # ---------- SIMULATED ----------
            grp_sim = _group_simulated(step_length, offset)
            try:
                sim = xr.open_zarr(store, group=grp_sim)
            except Exception:
                continue

            for needed in ("position", "frame", "track_start", "track_count", "target_step_index"):
                if needed not in sim:
                    warnings.warn(f"{store}/{grp_sim}: missing '{needed}', skipping simulated features.")
                    sim = None
                    break
            if (obs is None) or (sim is None):
                continue

            # observed references (for step start points)
            pos_obs = obs["position"].values
            fr_obs  = obs["frame"].values
            st_obs  = obs["track_start"].values.astype(np.int64)
            ct_obs  = obs["track_count"].values.astype(np.int64)

            # simulated
            pos_sim = sim["position"].values
            fr_sim  = sim["frame"].values
            tsi     = sim["target_step_index"].values.astype(np.int64)
            st_sim  = sim["track_start"].values.astype(np.int64)
            ct_sim  = sim["track_count"].values.astype(np.int64)
            n_tracks = sim.sizes["track"]

            npt_sim = pos_sim.shape[0]
            s_angle = np.full(npt_sim, np.nan, np.float32)
            s_dist_end = np.full(npt_sim, np.nan, np.float32)
            s_delta = np.full(npt_sim, np.nan, np.float32)
            s_len = np.full(npt_sim, np.nan, np.float32)
            s_dur = np.full(npt_sim, np.nan, np.float32)
            s_spd = np.full(npt_sim, np.nan, np.float32)
            s_head = np.full(npt_sim, np.nan, np.float32)  # <- heading_deg

            for ti in range(n_tracks):
                sO = st_obs[ti]; cO = ct_obs[ti]
                if cO < 2: continue
                ptsO = pos_obs[sO:sO+cO, :]
                frO  = fr_obs[sO:sO+cO]

                sS = st_sim[ti]; cS = ct_sim[ti]
                if cS == 0: continue

                end_xy = pos_sim[sS:sS+cS, :]
                end_fr = fr_sim[sS:sS+cS]
                tstep  = tsi[sS:sS+cS]

                valid = (tstep >= 0) & (tstep < (cO - 1))
                if not np.any(valid):
                    continue

                idx = np.nonzero(valid)[0]
                j = tstep[idx]                 # (M,)
                start_xy = ptsO[j, :]
                start_fr = frO[j]

                dx = end_xy[idx, 0] - start_xy[:, 0]
                dy = end_xy[idx, 1] - start_xy[:, 1]
                step_len = np.hypot(dx, dy).astype(np.float32)
                step_dur = ((end_fr[idx] - start_fr) / float(fps)).astype(np.float32)
                with np.errstate(divide="ignore", invalid="ignore"):
                    step_spd = (step_len / step_dur).astype(np.float32)

                dist_end = np.hypot(end_xy[idx, 0] - ox, end_xy[idx, 1] - oy).astype(np.float32)
                dist_start = np.hypot(start_xy[:, 0] - ox, start_xy[:, 1] - oy).astype(np.float32)
                delta = (dist_end - dist_start).astype(np.float32)

                ang = _angle_deg_between(dx, dy, (ox - start_xy[:, 0]), (oy - start_xy[:, 1])).astype(np.float32)
                heading_deg = (np.degrees(np.arctan2(dy, dx))).astype(np.float32)

                s_angle[sS + idx] = ang
                s_dist_end[sS + idx] = dist_end
                s_delta[sS + idx] = delta
                s_len[sS + idx] = step_len
                s_dur[sS + idx] = step_dur
                s_spd[sS + idx] = step_spd
                s_head[sS + idx] = heading_deg

            ds_sim_feats = xr.Dataset(
                data_vars={
                    "angle_to_observer": (("point",), s_angle),
                    "dist_to_observer": (("point",), s_dist_end),
                    "delta_observer_dist": (("point",), s_delta),
                    "step_length_m": (("point",), s_len),
                    "step_duration_s": (("point",), s_dur),
                    "step_speed_mps": (("point",), s_spd),
                    "heading_deg": (("point",), s_head),
                },
                coords={"point": np.arange(npt_sim)},
            )

            if chunks_point:
                ds_sim_feats = ds_sim_feats.chunk({"point": chunks_point})

            _set_var_attrs(ds_sim_feats, is_sim=True)
            ds_sim_feats.to_zarr(store, mode="a", group=grp_sim)

# ----------
# --- Step 5: Get zebra heights for visibility calculations ---
# ----------

def annotate_with_observer_height(
    step_length: float,
    offsets: list[float],
    rasters_directory: str | Path,
    zarr_root: str | Path = "tracks_xarray",
    obs_to_process: str | list[str] | None = None,
    var_name: str = "observer_height",
    show_progress: bool = True,
    chunks_point: int | None = None,  # keep existing chunking if None
):
    """
    Sample the precomputed 'zebra heights' raster for each step endpoint and write the
    values into both observed and simulated groups as <var_name> (default 'observer_height').

    - Reads store attrs['big_map'] to locate: <rasters_directory>/zebra_heights/{big_map}_ZebraHeights_1-5m.tif
    - Writes variable with attrs: units='m', description='Zebra head relative height (>= -inf, <= 1.5)'
    - Out-of-bounds or raster NoData -> NaN
    """
    if not HAVE_RASTERIO:
        raise RuntimeError("rasterio is required for Step 5. Please install rasterio.")

    zroot = Path(zarr_root)
    stores = _discover_stores(zroot, obs_to_process)

    obs_iter = tqdm(stores, desc="Step 5: observations", unit="obs") if show_progress else stores
    for store in obs_iter:
        root_ds = xr.open_zarr(store)
        map_name = root_ds.attrs.get("big_map", None)
        if not map_name:
            warnings.warn(f"{store.stem}: 'big_map' not found in attrs; skipping.")
            continue

        tif = Path(rasters_directory) / "zebra_heights" / f"{map_name}_ZebraHeights_1-5m.tif"
        if not tif.exists():
            warnings.warn(f"{store.stem}: zebra height raster not found: {tif}")
            continue

        with rio.open(tif) as src:
            off_iter = tqdm(offsets, desc=f"{store.stem}: offsets", leave=False) if show_progress else offsets
            for offset in off_iter:
                for grp_fn, is_sim in ((_group_observed, False), (_group_simulated, True)):
                    grp = grp_fn(step_length, offset)
                    try:
                        ds = xr.open_zarr(store, group=grp)
                    except Exception:
                        # group might not exist (e.g., simulated not generated yet)
                        continue

                    # Extract all endpoint coordinates
                    x, y = _get_xy_arrays(ds)
                    vals = _sample_raster_at_xy(src, x, y)  # float32 with NaNs

                    out = xr.Dataset({var_name: (("point",), vals)},
                                     coords={"point": np.arange(vals.size)})
                    # optional chunking
                    if chunks_point:
                        out = out.chunk({"point": chunks_point})

                    # add units/description
                    out[var_name].attrs.update({
                        "units": "m",
                        "description": "Zebra head relative height vs DSM (+1.5 ~ fully exposed; negative under canopy/terrain).",
                        "source_raster": str(tif),
                        "note": "NaN indicates out-of-bounds or NoData in raster.",
                    })

                    # Write alongside existing variables
                    out.to_zarr(store, mode="a", group=grp)


# ----------
# --- Step 6: Get step slope ---
# ----------

def annotate_with_ground_slope(
    step_length: float,
    offsets: list[float],
    rasters_directory: str | Path,
    zarr_root: str | Path = "tracks_xarray",
    obs_to_process: str | list[str] | None = None,
    write_alt_components: bool = True,   # also write start_alt_m, end_alt_m, alt_change_m
    chunks_point: int | None = None,
    show_progress: bool = True,
):
    """
    Step 6: sample DTM at start/end, reuse `step_length_m` (from Step 4), and write:
      - ground_slope_deg  (degree) = atan(Δalt / step_length_m)
      - ground_slope_pct  (%)      = 100 * Δalt / step_length_m
    Optionally also writes:
      - start_alt_m, end_alt_m, alt_change_m
    """
    if not HAVE_RASTERIO:
        raise RuntimeError("rasterio is required for Step 6. Please install rasterio.")

    zroot = Path(zarr_root)
    stores = _discover_stores(zroot, obs_to_process)

    var_attrs = {
        "ground_slope_deg": {"units": "degree",
                             "description": "Terrain slope along step: atan(alt_change_m / step_length_m) in degrees."},
        "ground_slope_pct": {"units": "%",
                             "description": "Percent grade: 100 * alt_change_m / step_length_m."},
        "start_alt_m": {"units": "m", "description": "DTM altitude at step start point."},
        "end_alt_m": {"units": "m", "description": "DTM altitude at step end point."},
        "alt_change_m": {"units": "m", "description": "DTM altitude change along step: end - start."},
    }

    obs_iter = tqdm(stores, desc="Step 6: observations", unit="obs") if show_progress else stores
    for store in obs_iter:
        root = xr.open_zarr(store)
        map_name = root.attrs.get("big_map", None)
        if not map_name:
            warnings.warn(f"{store.stem}: 'big_map' not found; skipping.")
            continue

        dtm_path = Path(rasters_directory) / "DTMS" / f"{map_name}_dtm.tif"
        if not dtm_path.exists():
            warnings.warn(f"{store.stem}: DTM not found: {dtm_path}")
            continue

        with rio.open(dtm_path) as src:
            off_iter = tqdm(offsets, desc=f"{store.stem}: offsets", leave=False) if show_progress else offsets
            for offset in off_iter:

                # ---------- OBSERVED ----------
                grp_obs = _group_observed(step_length, offset)
                try:
                    obs = xr.open_zarr(store, group=grp_obs)
                except Exception:
                    obs = None

                if obs is not None:
                    for needed in ("position", "track_start", "track_count", "step_length_m"):
                        if needed not in obs:
                            warnings.warn(f"{store}/{grp_obs}: missing '{needed}', skipping observed.")
                            obs = None
                            break

                if obs is not None:
                    x, y = _get_xy_arrays(obs)
                    alt_all = _sample_raster_at_xy(src, x, y)  # alt at every observed point

                    starts = obs["track_start"].values.astype(np.int64)
                    counts = obs["track_count"].values.astype(np.int64)
                    npt = alt_all.size

                    start_alt = np.full(npt, np.nan, np.float32)
                    end_alt   = np.full(npt, np.nan, np.float32)
                    d_alt     = np.full(npt, np.nan, np.float32)

                    # fill only at end points (s+1..s+c-1)
                    for s, c in zip(starts, counts):
                        if c < 2: continue
                        idx_end   = np.arange(s+1, s+c, dtype=np.int64)
                        idx_start = idx_end - 1
                        a_start = alt_all[idx_start]
                        a_end   = alt_all[idx_end]
                        start_alt[idx_end] = a_start.astype(np.float32)
                        end_alt[idx_end]   = a_end.astype(np.float32)
                        d_alt[idx_end]     = (a_end - a_start).astype(np.float32)

                    # reuse step_length_m (defined at end points)
                    sl = obs["step_length_m"].values.astype(np.float32, copy=False)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        slope_deg = np.degrees(np.arctan(d_alt / sl)).astype(np.float32)
                        slope_pct = (100.0 * d_alt / sl).astype(np.float32)

                    ds_out = xr.Dataset(
                        data_vars={
                            "ground_slope_deg": (("point",), slope_deg),
                            "ground_slope_pct": (("point",), slope_pct),
                            **({"start_alt_m": (("point",), start_alt),
                                "end_alt_m":   (("point",), end_alt),
                                "alt_change_m":(("point",), d_alt)} if write_alt_components else {})
                        },
                        coords={"point": np.arange(npt)},
                    )
                    for v, a in var_attrs.items():
                        if v in ds_out: ds_out[v].attrs.update(a)
                    if chunks_point:
                        ds_out = ds_out.chunk({"point": chunks_point})

                    ds_out.to_zarr(store, mode="a", group=grp_obs)

                # ---------- SIMULATED ----------
                grp_sim = _group_simulated(step_length, offset)
                try:
                    sim = xr.open_zarr(store, group=grp_sim)
                except Exception:
                    continue

                for needed in ("position", "track_start", "track_count", "target_step_index", "step_length_m"):
                    if needed not in sim:
                        warnings.warn(f"{store}/{grp_sim}: missing '{needed}', skipping simulated.")
                        sim = None
                        break
                if (obs is None) or (sim is None):
                    continue

                # end altitudes at simulated endpoints
                xS, yS = _get_xy_arrays(sim)
                alt_end_all = _sample_raster_at_xy(src, xS, yS)

                # observed starts to map by target_step_index
                pos_obs = obs["position"].values
                st_obs  = obs["track_start"].values.astype(np.int64)
                ct_obs  = obs["track_count"].values.astype(np.int64)

                st_sim = sim["track_start"].values.astype(np.int64)
                ct_sim = sim["track_count"].values.astype(np.int64)
                tsi    = sim["target_step_index"].values.astype(np.int64)
                nptS   = alt_end_all.size

                start_alt = np.full(nptS, np.nan, np.float32)
                end_alt   = alt_end_all.astype(np.float32, copy=True)
                d_alt     = np.full(nptS, np.nan, np.float32)

                # map simulated points to their observed start points
                for ti in range(sim.sizes["track"]):
                    sO = st_obs[ti]; cO = ct_obs[ti]
                    if cO < 2: continue
                    sS = st_sim[ti]; cS = ct_sim[ti]
                    if cS == 0: continue

                    idx_block = np.arange(sS, sS + cS, dtype=np.int64)
                    tstep = tsi[idx_block]
                    valid = (tstep >= 0) & (tstep < (cO - 1))
                    if not np.any(valid): continue

                    idxv = idx_block[valid]
                    j = tstep[valid]
                    start_pts = pos_obs[sO + j, :]
                    a_start = _sample_raster_at_xy(src, start_pts[:, 0], start_pts[:, 1])
                    start_alt[idxv] = a_start.astype(np.float32)
                    d_alt[idxv] = (end_alt[idxv] - start_alt[idxv]).astype(np.float32)

                sl = sim["step_length_m"].values.astype(np.float32, copy=False)
                with np.errstate(divide="ignore", invalid="ignore"):
                    slope_deg = np.degrees(np.arctan(d_alt / sl)).astype(np.float32)
                    slope_pct = (100.0 * d_alt / sl).astype(np.float32)

                ds_outS = xr.Dataset(
                    data_vars={
                        "ground_slope_deg": (("point",), slope_deg),
                        "ground_slope_pct": (("point",), slope_pct),
                        **({"start_alt_m": (("point",), start_alt),
                            "end_alt_m":   (("point",), end_alt),
                            "alt_change_m":(("point",), d_alt)} if write_alt_components else {})
                    },
                    coords={"point": np.arange(nptS)},
                )
                for v, a in var_attrs.items():
                    if v in ds_outS: ds_outS[v].attrs.update(a)
                if chunks_point:
                    ds_outS = ds_outS.chunk({"point": chunks_point})

                ds_outS.to_zarr(store, mode="a", group=grp_sim)

# ----------
# --- Step 7: Get social features ---
# ----------

def annotate_social_features(
    step_length: float,
    offsets: list[float],
    rasters_directory: str | Path,
    zarr_root: str | Path = "tracks_xarray",
    obs_to_process: str | list[str] | None = None,
    social_radius: float = 10.0,          # meters: density radius
    compute_visibility: bool = True,      # toggle expensive LOS calls
    visibility_max_range: float | None = None,  # only compute LOS for neighbors within this range (meters)
    max_neighbors_los: int | None = None, # cap count of neighbors to test LOS (nearest first)
    write_neighbor_table: bool = True,    # write ragged neighbor arrays
    show_progress: bool = True,
    chunks_point: int | None = None,
):
    """
    Step 7: per-step social features (observed & simulated).

    Writes per-point vars:
      - social_density (int32)         : # neighbors within `social_radius`
      - social_visible_count (int32)   : # neighbors with line-of-sight True (if computed)
      - nn_distance_m (float32)        : distance to nearest neighbor
      - nn_visible (int8)              : 1 if nearest neighbor visible else 0 (NaN if not computed)
      - nn_track (int32)               : track index of nearest neighbor (-1 if none)
      - nn_individual_ID (string)      : nearest neighbor ID (if available)
      - nn_species (string)            : nearest neighbor species
      - nn_age (string)                : nearest neighbor age class

    Optionally writes a ragged neighbor table:
      - neighbors_start(point), neighbors_count(point)
      - neighbor_track(neighbor), neighbor_distance_m(neighbor), neighbor_visible(neighbor)
      - neighbor_position(neighbor, location)
    """
    if not HAVE_RASTERIO:
        raise RuntimeError("rasterio is required.")
    if compute_visibility and not HAVE_GDAL:
        warnings.warn("GDAL not available; compute_visibility=False will be used.")
        compute_visibility = False

    zroot = Path(zarr_root)
    stores = _discover_stores(zroot, obs_to_process)

    obs_iter = tqdm(stores, desc="Step 7: observations", unit="obs") if show_progress else stores
    for store in obs_iter:
        root = xr.open_zarr(store)

        # root: raw tracks (track, frame, location) + metadata
        if not all(k in root for k in ("position", "location")):
            warnings.warn(f"{store.stem}: root missing 'position'/'location'; skipping.")
            continue
        x_all, y_all = _xy_from_root(root)      # (track, frame)
        n_tracks, n_frames = x_all.shape

        # track-level metadata (optional but recommended)
        trk_id = root.get("individual_ID")
        trk_sp = root.get("species")
        trk_age = root.get("age")

        # map raster names
        big_map = root.attrs.get("big_map", None)
        if not big_map:
            warnings.warn(f"{store.stem}: 'big_map' not found; skipping.")
            continue

        dsm_path = Path(rasters_directory) / "DSMs" / f"{big_map}_dsm.tif"
        zeb_path = Path(rasters_directory) / "zebra_heights" / f"{big_map}_ZebraHeights_1-5m.tif"
        if not (dsm_path.exists() and zeb_path.exists()):
            warnings.warn(f"{store.stem}: DSM or zebra height raster missing.")
            continue

        # open rasters once per observation
        dsm_rio = rio.open(dsm_path)
        zeb_rio = rio.open(zeb_path)
        # GDAL band for LOS (DSM)
        gdal_band = None
        if compute_visibility:
            ds_gdal = gdal.Open(str(dsm_path))
            gdal_band = ds_gdal.GetRasterBand(1)

        try:
            off_iter = tqdm(offsets, desc=f"{store.stem}: offsets", leave=False) if show_progress else offsets
            for offset in off_iter:
                for grp_fn, is_sim in ((_group_observed, False), (_group_simulated, True)):
                    grp = grp_fn(step_length, offset)
                    try:
                        ds = xr.open_zarr(store, group=grp)
                    except Exception:
                        continue

                    # need: position(point, location), frame(point), track_start/count
                    for req in ("position", "frame", "track_start", "track_count"):
                        if req not in ds:
                            warnings.warn(f"{store.stem}/{grp}: missing '{req}', skipping.")
                            ds = None
                            break
                    if ds is None:
                        continue

                    # prefer to reuse 'observer_height' (Step 5) for focal altitude; else sample on-the-fly
                    have_focal_h = "observer_height" in ds

                    pos_pt = ds["position"].values           # (point, 2)
                    frames_pt = ds["frame"].values           # (point,) float
                    starts = ds["track_start"].values.astype(np.int64)
                    counts = ds["track_count"].values.astype(np.int64)
                    npt = pos_pt.shape[0]

                    # allocate outputs (per-point)
                    social_density = np.zeros(npt, np.int32)
                    social_visible = np.full(npt, 0, np.int32)
                    nn_dist = np.full(npt, np.nan, np.float32)
                    nn_vis  = np.full(npt, np.nan, np.float32)  # 1/0 (store as float NaN-able)
                    nn_track = np.full(npt, -1, np.int32)
                    nn_id   = np.empty(npt, dtype=object) if trk_id is not None else None
                    nn_sp   = np.empty(npt, dtype=object) if trk_sp is not None else None
                    nn_age  = np.empty(npt, dtype=object) if trk_age is not None else None

                    # ragged neighbor arrays (if enabled)
                    neigh_start = np.zeros(npt, np.int64) if write_neighbor_table else None
                    neigh_count = np.zeros(npt, np.int64) if write_neighbor_table else None
                    neigh_track = []   # flattened
                    neigh_dist  = []
                    neigh_vis   = []
                    neigh_pos   = []   # (x,y) flattened

                    # iterate tracks (ragged)
                    for ti, (s, c) in enumerate(zip(starts, counts)):
                        if c < 2 and not is_sim:
                            # observed: first point has no step; but we still annotate at all endpoints
                            pass
                        if c == 0:
                            continue

                        # all points in this track block
                        idx_block = np.arange(s, s + c, dtype=np.int64)
                        # we annotate at ALL points (observed: endpoints carry step-based vars already)
                        for pi in idx_block:
                            fx, fy = pos_pt[pi, 0], pos_pt[pi, 1]
                            f_end = frames_pt[pi]
                            if not np.isfinite(f_end):
                                continue
                            fr = int(round(float(f_end)))
                            if fr < 0 or fr >= n_frames:
                                continue

                            # focal altitude (DSM + head height)
                            if have_focal_h:
                                focal_h = float(ds["observer_height"].values[pi])
                            else:
                                focal_h = float(_sample_raster_at_xy(zeb_rio, np.array([fx]), np.array([fy]))[0])
                            # DSM altitude at focal pixel
                            rA, cA = _rowcol(dsm_rio, fx, fy)
                            if (rA < 0 or cA < 0 or rA >= dsm_rio.height or cA >= dsm_rio.width):
                                # OOB → skip social metrics (leave NaNs / zeros)
                                continue
                            # extract ground elevation at focal
                            z_ground_A = dsm_rio.read(1, window=((rA, rA+1), (cA, cA+1))).astype(np.float32)[0,0]
                            zA = z_ground_A + focal_h

                            # neighbors at this frame from ROOT raw tracks
                            xs = x_all[:, fr]  # (track,)
                            ys = y_all[:, fr]
                            valid = np.isfinite(xs) & np.isfinite(ys)
                            valid[ti] = False   # exclude focal track
                            if not valid.any():
                                continue

                            dx = xs[valid] - fx
                            dy = ys[valid] - fy
                            d = np.hypot(dx, dy)
                            tr_idx = np.nonzero(valid)[0]

                            # density within radius
                            social_density[pi] = int(np.sum(d < social_radius))

                            # nearest neighbor
                            if d.size > 0:
                                k = int(np.argmin(d))
                                nn_dist[pi] = float(d[k])
                                nn_track[pi] = int(tr_idx[k])
                                # resolve metadata (strings)
                                if nn_id is not None:
                                    try: nn_id[pi] = str(trk_id.values[nn_track[pi]])
                                    except Exception: nn_id[pi] = None
                                if nn_sp is not None:
                                    try: nn_sp[pi] = str(trk_sp.values[nn_track[pi]])
                                    except Exception: nn_sp[pi] = None
                                if nn_age is not None:
                                    try: nn_age[pi] = str(trk_age.values[nn_track[pi]])
                                    except Exception: nn_age[pi] = None

                            # visibility?
                            vis_flags = None
                            if compute_visibility and d.size > 0:
                                # decide which neighbors to test
                                idx_order = np.argsort(d)  # nearest first
                                cand_idx = idx_order
                                if visibility_max_range is not None:
                                    cand_idx = cand_idx[d[idx_order] <= visibility_max_range]
                                if max_neighbors_los is not None and cand_idx.size > max_neighbors_los:
                                    cand_idx = cand_idx[:max_neighbors_los]

                                vis_flags = np.zeros(d.size, dtype=bool)
                                if cand_idx.size > 0:
                                    # pre-read DSM array tile lazily via rowcol per neighbor
                                    # (fine for tens of neighbors)
                                    for kk in cand_idx.tolist():
                                        nx, ny = xs[valid][kk], ys[valid][kk]
                                        # neighbor head height
                                        nh = float(_sample_raster_at_xy(zeb_rio, np.array([nx]), np.array([ny]))[0])
                                        rB, cB = _rowcol(dsm_rio, nx, ny)
                                        if (rB < 0 or cB < 0 or rB >= dsm_rio.height or cB >= dsm_rio.width):
                                            vis = False
                                        else:
                                            z_ground_B = dsm_rio.read(1, window=((rB, rB+1), (cB, cB+1))).astype(np.float32)[0,0]
                                            zB = z_ground_B + nh
                                            vis = _los_visible(gdal_band, cA, rA, zA, cB, rB, zB) if compute_visibility else False
                                        vis_flags[kk] = vis

                                # total visible
                                social_visible[pi] = int(vis_flags.sum())
                                # nearest-neighbor visible?
                                if d.size > 0:
                                    nn_vis[pi] = float(vis_flags[np.argmin(d)])
                            else:
                                # no LOS computed: leave nn_vis NaN, social_visible 0
                                pass

                            # ragged neighbor table
                            if write_neighbor_table:
                                start = len(neigh_track)
                                neigh_start[pi] = start
                                cnt = int(d.size)
                                neigh_count[pi] = cnt
                                if cnt:
                                    neigh_track.extend(tr_idx.astype(np.int32).tolist())
                                    neigh_dist.extend(d.astype(np.float32).tolist())
                                    if vis_flags is None:
                                        neigh_vis.extend([False]*cnt)
                                    else:
                                        neigh_vis.extend(vis_flags.astype(bool).tolist())
                                    # positions as (x,y)
                                    neigh_pos.extend(np.stack([xs[valid], ys[valid]], axis=1).tolist())

                    # pack outputs to Dataset
                    data_vars = {
                        "social_density": (("point",), social_density.astype(np.int32)),
                        "social_visible_count": (("point",), social_visible.astype(np.int32)),
                        "nn_distance_m": (("point",), nn_dist),
                        "nn_visible": (("point",), nn_vis),  # 1.0/0.0/NaN
                        "nn_track": (("point",), nn_track),
                    }
                    # strings (optional)
                    if nn_id is not None:
                        data_vars["nn_individual_ID"] = (("point",), nn_id)
                    if nn_sp is not None:
                        data_vars["nn_species"] = (("point",), nn_sp)
                    if nn_age is not None:
                        data_vars["nn_age"] = (("point",), nn_age)

                    ds_out = xr.Dataset(data_vars, coords={"point": np.arange(npt)})
                    # attrs
                    ds_out["social_density"].attrs.update({"units":"count", "description": f"# neighbors within {social_radius} m."})
                    ds_out["social_visible_count"].attrs.update({"units":"count", "description":"# neighbors with line-of-sight True (if computed)."})
                    ds_out["nn_distance_m"].attrs.update({"units":"m", "description":"Distance to nearest neighbor."})
                    ds_out["nn_visible"].attrs.update({"units":"1", "description":"Nearest neighbor line-of-sight (1=True, 0=False, NaN=not computed)."})
                    ds_out["nn_track"].attrs.update({"units":"index", "description":"Track index of nearest neighbor (-1 if none)."})
                    if "nn_individual_ID" in ds_out: ds_out["nn_individual_ID"].attrs["description"] = "Nearest neighbor individual ID."
                    if "nn_species" in ds_out: ds_out["nn_species"].attrs["description"] = "Nearest neighbor species."
                    if "nn_age" in ds_out: ds_out["nn_age"].attrs["description"] = "Nearest neighbor age class."

                    if chunks_point:
                        ds_out = ds_out.chunk({"point": chunks_point})

                    ds_out.to_zarr(store, mode="a", group=grp)

                    # ragged neighbor table
                    if write_neighbor_table:
                        if len(neigh_track) == 0:
                            # create empty structure
                            ds_nb = xr.Dataset(
                                data_vars={
                                    "neighbors_start": (("point",), neigh_start),
                                    "neighbors_count": (("point",), neigh_count),
                                    "neighbor_track": (("neighbor",), np.zeros(0, np.int32)),
                                    "neighbor_distance_m": (("neighbor",), np.zeros(0, np.float32)),
                                    "neighbor_visible": (("neighbor",), np.zeros(0, np.int8)),
                                    "neighbor_position": (("neighbor","location"), np.zeros((0,2), np.float32)),
                                },
                                coords={"point": np.arange(npt), "neighbor": np.arange(0), "location": ["easting","northing"]}
                            )
                        else:
                            neigh_track_arr = np.asarray(neigh_track, dtype=np.int32)
                            neigh_dist_arr  = np.asarray(neigh_dist,  dtype=np.float32)
                            neigh_vis_arr   = np.asarray(neigh_vis,   dtype=bool).astype(np.int8)
                            neigh_pos_arr   = np.asarray(neigh_pos,   dtype=np.float32)  # (N,2)

                            ds_nb = xr.Dataset(
                                data_vars={
                                    "neighbors_start": (("point",), neigh_start),
                                    "neighbors_count": (("point",), neigh_count),
                                    "neighbor_track": (("neighbor",), neigh_track_arr),
                                    "neighbor_distance_m": (("neighbor",), neigh_dist_arr),
                                    "neighbor_visible": (("neighbor",), neigh_vis_arr),
                                    "neighbor_position": (("neighbor","location"), neigh_pos_arr),
                                },
                                coords={"point": np.arange(npt), "neighbor": np.arange(neigh_track_arr.size),
                                        "location": ["easting","northing"]}
                            )
                        # helpful attrs
                        ds_nb["neighbors_start"].attrs["description"] = "Start index into flattened neighbor arrays for each point."
                        ds_nb["neighbors_count"].attrs["description"] = "Number of neighbors for each point."
                        ds_nb["neighbor_track"].attrs["description"] = "Neighbor track index (map to root track metadata)."
                        ds_nb["neighbor_distance_m"].attrs.update({"units":"m", "description":"Distance to each neighbor."})
                        ds_nb["neighbor_visible"].attrs["description"] = "Line-of-sight visibility to each neighbor (1/0)."
                        ds_nb["neighbor_position"].attrs.update({"units":"m", "description":"Neighbor XY at focal frame (UTM meters)."})

                        if chunks_point:
                            ds_nb = ds_nb.chunk({"point": chunks_point})
                        ds_nb.to_zarr(store, mode="a", group=f"{grp}/social_neighbors")

        finally:
            try: dsm_rio.close()
            except Exception: pass
            try: zeb_rio.close()
            except Exception: pass
            if compute_visibility and gdal_band is not None:
                try:
                    # close GDAL dataset
                    gdal_band = None
                    ds_gdal = None
                except Exception:
                    pass

# ----------
# --- Step 8: Get ground cover ---
# ----------
def annotate_ground_cover(
    step_length: float,
    offsets: list[float],
    rasters_directory: str | Path,
    *,
    zarr_root: str | Path = "tracks_xarray",
    obs_to_process: str | list[str] | None = None,
    class_raster_subdir: str = "ground_classification",
    class_map: dict[int, str] | None = None,
    write_names: bool = False,
    show_progress: bool = True,
    chunks_point: int | None = None,
):
    """
    Step 8: sample ground-class raster at step endpoints for selected observations.

    obs_to_process:
      - None -> process all observation*.zarr under zarr_root
      - "ob015" / "observation015" / ["ob015","ob074", ...] are all accepted
    """
    zroot = Path(zarr_root)
    stores = _discover_stores(zroot, obs_to_process)
    if not stores:
        raise FileNotFoundError(f"No matching observation*.zarr in {zroot} (selection={obs_to_process})")

    def _open_group(store: Path, group: str) -> xr.Dataset | None:
        try:
            return xr.open_zarr(store, group=group)
        except Exception:
            return None

    def _sample_raster(src: rio.io.DatasetReader, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        out = np.full(xs.shape[0], np.nan, dtype=np.float32)
        b = src.bounds
        inb = (xs >= b.left) & (xs <= b.right) & (ys >= b.bottom) & (ys <= b.top)
        if inb.any():
            coords = list(zip(xs[inb], ys[inb]))
            arr = np.fromiter((v[0] for v in src.sample(coords)), dtype=np.float32, count=inb.sum())
            if src.nodata is not None:
                arr = np.where(arr == src.nodata, np.nan, arr)
            out[inb] = arr
        return out

    def _to_fixed_unicode(arr_like: np.ndarray) -> np.ndarray:
        vals = np.asarray(arr_like, dtype=object)
        maxlen = int(max((len(str(s)) for s in vals if s is not None), default=0))
        return np.array([("" if v is None else str(v)) for v in vals], dtype=f"<U{maxlen}")

    obs_iter = tqdm(stores, desc="Step 8: observations", unit="obs") if show_progress else stores
    for store in obs_iter:
        root = xr.open_zarr(store)
        # derive obXXX from observation_id (e.g., observation015 -> ob015)
        obs_id = root.attrs.get("observation_id", store.stem)
        m = re.match(r"^observation(\d+)$", str(obs_id), flags=re.IGNORECASE)
        ob_id = f"ob{m.group(1)}" if m else str(obs_id).replace("observation", "ob")

        gc_dir = Path(rasters_directory) / class_raster_subdir
        gc_path = gc_dir / f"{ob_id}_groundclass_50cm.tif"
        if not gc_path.exists():
            warnings.warn(f"{obs_id}: ground-class raster not found at {gc_path}; skipping.")
            continue

        with rio.open(gc_path) as gc:
            off_iter = tqdm(offsets, desc=f"{store.stem}: offsets", leave=False) if show_progress else offsets
            for offset in off_iter:
                for grp, _is_sim in ((_group_observed(step_length, offset), False),
                                     (_group_simulated(step_length, offset), True)):
                    ds = _open_group(store, grp)
                    if ds is None or "position" not in ds:
                        continue

                    pos = ds["position"].values  # (point,2) -> [x,y]
                    xs = pos[:, 0].astype(np.float64)
                    ys = pos[:, 1].astype(np.float64)

                    vals = _sample_raster(gc, xs, ys)  # float32 with NaN for nodata/OOB
                    codes = np.where(np.isfinite(vals), vals.astype(np.int16), -1).astype(np.int16)

                    data_vars = {"ground_class": (("point",), codes)}
                    ds_out = xr.Dataset(data_vars, coords={"point": np.arange(codes.shape[0])})
                    ds_out["ground_class"].attrs.update({
                        "description": "Ground cover class code at step end point.",
                        "missing_value": -1,
                        "source_raster": str(gc_path),
                    })
                    if class_map:
                        ds_out["ground_class"].attrs["class_map"] = {int(k): str(v) for k, v in class_map.items()}

                    if write_names and class_map:
                        names = np.array([class_map.get(int(c), None) if c >= 0 else None for c in codes], dtype=object)
                        ds_out["ground_class_name"] = (("point",), _to_fixed_unicode(names))
                        ds_out["ground_class_name"].attrs["description"] = "Human-readable class name."

                    if chunks_point:
                        ds_out = ds_out.chunk({"point": chunks_point})

                    ds_out.to_zarr(store, group=grp, mode="a")


# ----------
# --- Step 9: Generate viewsheds and annotate visibility ---
# ----------

def annotate_viewsheds(
    step_length: float,
    offsets: list[float],
    rasters_directory: str | Path,
    viewshed_save_directory: str | Path,
    *,
    zarr_root: str | Path = "tracks_xarray",
    obs_to_process: str | list[str] | None = None,
    radius: float = 30.0,
    threads: int = 8,
    dataset_tag: str | None = None,
    # independent toggles:
    keep_observed: bool = True,
    keep_simulated: bool = False,
    # reuse behavior:
    reuse_observed: bool = True,
    reuse_simulated: bool = False,
    # progress:
    show_progress: bool = True,
    progress_mode: str = "observation",  # "observation" | "group" | "none"
    # NEW: on-the-fly DSM downsampling
    target_cell_size_m: float | None = None,     # e.g., 0.05 for 5 cm, 0.10 for 10 cm
    resample_alg: str = "bilinear",              # "nearest","bilinear","cubic","average",...
):
    """
    Step 10 (optimized MEM-first): compute/read viewsheds and write `viewshed_vis`.

    - If a TIFF exists and reuse_* is True -> read mean from file, skip compute.
    - Else compute once in memory; then persist based on keep_observed/keep_simulated.
    - Downsample DSM (optional) once per observation into a MEM dataset for significant speedups.
    - Per-track traversal (no 'track_unknown'); folders are observation/trackNNN/...
    - Filenames use frame numbers; simulated also include replicate.
    """
    # dataset tag for simulated folders (e.g., "steps_5m")
    def _dist_tag_local(x: float) -> str:
        x = float(x);  return f"{int(x)}m" if x.is_integer() else f"{str(x).replace('.', 'p')}m"
    auto_dataset_tag = dataset_tag or f"steps_{_dist_tag_local(step_length)}"

    zroot = Path(zarr_root)
    outroot = Path(viewshed_save_directory)
    stores = _discover_stores(zroot, obs_to_process)
    if not stores:
        raise FileNotFoundError(f"No matching observation*.zarr in {zroot} (selection={obs_to_process})")

    obs_iter = tqdm(stores, desc="Step 10: observations", unit="obs", dynamic_ncols=True, leave=False) \
               if (show_progress and progress_mode != "none") else stores

    for store in obs_iter:
        root = xr.open_zarr(store)
        obs_id = root.attrs.get("observation_id", store.stem)
        big_map = root.attrs.get("big_map", None)

        # Resolve DSM path
        dsm_dir = Path(rasters_directory) / "DSMs"
        dsm_path = None
        if big_map and (dsm_dir / f"{big_map}_dsm.tif").exists():
            dsm_path = dsm_dir / f"{big_map}_dsm.tif"
        elif (dsm_dir / f"{obs_id}_dsm.tif").exists():
            dsm_path = dsm_dir / f"{obs_id}_dsm.tif"
        if dsm_path is None:
            warnings.warn(f"{obs_id}: DSM not found; skipping.")
            continue

        # Build per-observation DSM handle (optionally downsampled)
        if target_cell_size_m is not None:
            dsm_handle = _build_downsampled_dsm_mem(dsm_path, target_cell_size_m, resample_alg=resample_alg)
        else:
            dsm_handle = gdal.Open(str(dsm_path))
            if dsm_handle is None:
                warnings.warn(f"{obs_id}: could not open DSM; skipping.")
                continue
        dsm_band = dsm_handle.GetRasterBand(1)

        # Build tasks list for progress sizing
        tasks = []
        for offset in offsets:
            for grp, is_sim in ((_group_observed(step_length, offset), False),
                                (_group_simulated(step_length, offset), True)):
                try:
                    ds = xr.open_zarr(store, group=grp)
                except Exception:
                    ds = None
                if ds is None or "position" not in ds or "observer_height" not in ds:
                    continue

                pos = ds["position"].values              # (point,2)
                frames = ds["frame"].values              # (point,)
                xs = pos[:, 0].astype(float)
                ys = pos[:, 1].astype(float)
                heights = ds["observer_height"].values.astype(float)

                # ragged by track
                if ("track_start" not in ds) or ("track_count" not in ds) or ("track" not in ds.coords):
                    warnings.warn(f"{obs_id}/{grp}: missing ragged track metadata; skipping group.")
                    continue
                starts = ds["track_start"].values.astype(np.int64)
                counts = ds["track_count"].values.astype(np.int64)
                n_tracks = ds.sizes["track"]

                # replicate only present for simulated
                repl = ds["replicate"].values.astype(int) if (is_sim and "replicate" in ds) else None

                tasks.append({
                    "grp": grp, "is_sim": is_sim, "offset": offset,
                    "xs": xs, "ys": ys, "heights": heights, "frames": frames,
                    "starts": starts, "counts": counts, "n_tracks": n_tracks,
                    "replicate": repl,
                })

        # observation-level progress bar
        pbar_obs = None
        if progress_mode == "observation" and show_progress:
            total_pts = int(sum(t["xs"].size for t in tasks))
            pbar_obs = tqdm(total=total_pts, desc=f"{obs_id}: viewsheds", unit="pts",
                            dynamic_ncols=True, leave=False)

        # Process tasks
        for t in tasks:
            grp = t["grp"]; is_sim = t["is_sim"]; offset = t["offset"]
            xs, ys, heights, frames = t["xs"], t["ys"], t["heights"], t["frames"]
            starts, counts, n_tracks = t["starts"], t["counts"], t["n_tracks"]
            replicate = t["replicate"]

            # per-group progress (optional)
            pbar_grp = None
            if progress_mode == "group" and show_progress:
                pbar_grp = tqdm(total=int(xs.size), desc=f"{obs_id} | {grp}", unit="pts",
                                dynamic_ncols=True, leave=False)

            vis = np.full(xs.shape[0], np.nan, dtype=np.float32)

            # iterate per track to get stable track labels
            for ti in range(n_tracks):
                s = int(starts[ti]); c = int(counts[ti])
                if c == 0:
                    continue
                track_label = f"track{ti:03d}"

                # point indices for this track block
                idx_block = np.arange(s, s + c, dtype=np.int64)

                # compute for each point
                for p in idx_block:
                    X, Y, hgt = float(xs[p]), float(ys[p]), float(heights[p])
                    frame_num = int(round(float(frames[p])))

                    # output file path
                    folder = _viewshed_folder(
                        viewshed_root=outroot,
                        observation_id=obs_id,
                        track_label=track_label,
                        is_simulated=is_sim,
                        dataset_tag=(auto_dataset_tag if is_sim else None),
                        offset=(offset if is_sim else None),
                    )
                    if is_sim:
                        rep = int(replicate[p]) if replicate is not None else 0
                        fname = _viewshed_filename_sim(obs_id, track_label, frame_num, rep, radius)
                    else:
                        fname = _viewshed_filename_observed(obs_id, track_label, frame_num, radius)
                    tif_path = folder / fname

                    # reuse logic
                    reuse = (reuse_simulated if is_sim else reuse_observed)
                    if reuse and tif_path.exists():
                        try:
                            ds_tif = gdal.Open(str(tif_path))
                            if ds_tif:
                                band = ds_tif.GetRasterBand(1)
                                stats = band.GetStatistics(0, 1)
                                if stats is not None:
                                    vis[p] = float(stats[2])
                                else:
                                    arr = band.ReadAsArray()
                                    nod = band.GetNoDataValue()
                                    if nod is not None:
                                        arr = np.where(arr == nod, np.nan, arr)
                                    vis[p] = float(np.nanmean(arr))
                                ds_tif = None
                                # progress & continue
                                if pbar_obs is not None: pbar_obs.update(1)
                                if pbar_grp is not None: pbar_grp.update(1)
                                continue
                        except Exception:
                            # fall through to compute
                            pass

                    # compute in memory once
                    try:
                        mem_ds = _viewshed_generate_mem_from_band(dsm_band, X, Y, hgt, radius_m=radius)
                        try:
                            vis[p] = _viewshed_mean_visible_from_mem(mem_ds)
                            # persist depending on class
                            if (keep_simulated if is_sim else keep_observed):
                                _save_mem_to_gtiff(mem_ds, tif_path, threads=threads)
                        finally:
                            mem_ds = None
                    except Exception as exc:
                        warnings.warn(f"{obs_id} {grp} track {ti} point {p}: viewshed error: {exc}")
                        vis[p] = np.nan

                    if pbar_obs is not None: pbar_obs.update(1)
                    if pbar_grp is not None: pbar_grp.update(1)

            if pbar_grp is not None:
                pbar_grp.close()

            # write visibilities back to the group (single write)
            out = xr.Dataset({"viewshed_vis": (("point",), vis.astype(np.float32))})
            out["viewshed_vis"].attrs.update({
                "description": f"Proportion visible in {int(radius)} m viewshed (1 visible, 0 invisible).",
                "radius_m": float(radius),
                "generator": "gdal.ViewshedGenerate (MEM path, optional GTiff persist, optional downsampled DSM)",
                "visibleVal": 1, "invisibleVal": 0, "nodataVal": -10000,
            })
            out.to_zarr(store, group=grp, mode="a")

        if pbar_obs is not None:
            pbar_obs.close()

        # release per-observation DSM handle
        dsm_band = None
        dsm_handle = None

# def annotate_viewsheds(
#     step_length: float,
#     offsets: list[float],
#     rasters_directory: str | Path,
#     viewshed_save_directory: str | Path,
#     *,
#     zarr_root: str | Path = "tracks_xarray",
#     obs_to_process: str | list[str] | None = None,
#     radius: float = 30.0,
#     threads: int = 8,
#     dataset_tag: str | None = None,
#     keep_observed: bool = True,
#     keep_simulated: bool = False,
#     reuse_observed: bool = True,
#     reuse_simulated: bool = False,
#     show_progress: bool = True,
#     progress_mode: str = "observation",  # "observation" | "group" | "none"
# ):
#     """
#     Step 10 (MEM-first): compute/read viewsheds and write `viewshed_vis`.

#     Fix: per-point track labeling to avoid 'track_unknown' and put files under the
#     correct <observation>/<track_label>/... folder.

#     - If a TIFF exists and reuse_* is True -> read mean from file, skip compute.
#     - Else compute once in memory; persist based on keep_observed/keep_simulated.
#     """
#     def _dist_tag(x: float) -> str:
#         x = float(x)
#         return f"{int(x)}m" if x.is_integer() else f"{str(x).replace('.', 'p')}m"

#     auto_dataset_tag = dataset_tag or f"steps_{_dist_tag(step_length)}"

#     zroot = Path(zarr_root)
#     outroot = Path(viewshed_save_directory)
#     stores = _discover_stores(zroot, obs_to_process)
#     if not stores:
#         raise FileNotFoundError(f"No matching observation*.zarr in {zroot} (selection={obs_to_process})")

#     obs_iter = tqdm(stores, desc="Step 10: observations", unit="obs", dynamic_ncols=True, leave=False) \
#                if (show_progress and progress_mode != "none") else stores

#     for store in obs_iter:
#         root = xr.open_zarr(store)
#         obs_id = root.attrs.get("observation_id", store.stem)
#         big_map = root.attrs.get("big_map", None)

#         # DSM path
#         dsm_dir = Path(rasters_directory) / "DSMs"
#         dsm_path = None
#         if big_map and (dsm_dir / f"{big_map}_dsm.tif").exists():
#             dsm_path = dsm_dir / f"{big_map}_dsm.tif"
#         elif (dsm_dir / f"{obs_id}_dsm.tif").exists():
#             dsm_path = dsm_dir / f"{obs_id}_dsm.tif"
#         if dsm_path is None:
#             warnings.warn(f"{obs_id}: DSM not found; skipping.")
#             continue

#         # Build tasks (collect arrays so we can size progress bars)
#         tasks = []
#         for offset in offsets:
#             for grp, is_sim in ((_group_observed(step_length, offset), False),
#                                 (_group_simulated(step_length, offset), True)):
#                 try:
#                     ds = xr.open_zarr(store, group=grp)
#                 except Exception:
#                     ds = None
#                 if ds is None or "position" not in ds or "observer_height" not in ds:
#                     continue

#                 # Core arrays
#                 pos = ds["position"].values  # (point, 2)
#                 xs = pos[:, 0].astype(float)
#                 ys = pos[:, 1].astype(float)
#                 heights = ds["observer_height"].values.astype(float)
#                 npt = xs.size

#                 # Ragged mapping: point -> track index
#                 if ("track_start" not in ds) or ("track_count" not in ds):
#                     warnings.warn(f"{obs_id}/{grp}: missing track_start/track_count; using 'track_unknown'.")
#                     point_track = np.full(npt, -1, dtype=np.int32)
#                     track_labels = ["track_unknown"]
#                 else:
#                     starts = ds["track_start"].values.astype(np.int64)
#                     counts = ds["track_count"].values.astype(np.int64)
#                     n_tracks = starts.size

#                     # Build per-track label list
#                     track_labels = []
#                     if "track" in ds.coords and ds["track"].sizes.get("track", n_tracks) == n_tracks:
#                         # If user already stored track labels, use them — but strip observation prefix if present
#                         tvals = ds["track"].values
#                         for ti in range(n_tracks):
#                             raw = str(tvals[ti])
#                             # Normalize:
#                             #   - if already "track000", use as is
#                             #   - if something like "observation015_track000", strip the observation part
#                             m = re.search(r"(track\d+)", raw)
#                             if m:
#                                 label = m.group(1)   # → track000
#                             else:
#                                 label = f"track{ti:03d}"
#                             track_labels.append(label)
#                     else:
#                         # No track coord — fall back to simple track000, track001, ...
#                         track_labels = [f"track{ti:03d}" for ti in range(n_tracks)]

#                     # Vectorized-ish map: for each track block, assign its track index to its points
#                     point_track = np.full(npt, -1, dtype=np.int32)
#                     for ti in range(n_tracks):
#                         s = int(starts[ti]); c = int(counts[ti])
#                         if c > 0:
#                             point_track[s:s+c] = ti

#                 # File IDs: prefer explicit 'id' var if present; else synthesize
#                 if "id" in ds:
#                     ids = [str(i) for i in ds["id"].values]
#                 else:
#                     # Use frame if available for determinism across discretizations; else fallback to pt index
#                     if "frame" in ds:
#                         frames = ds["frame"].values.astype(float)
#                         ids = []
#                         for i in range(npt):
#                             ti = int(point_track[i]) if point_track[i] >= 0 else 0
#                             label = track_labels[ti] if (0 <= ti < len(track_labels)) else "track_unknown"
#                             frame_i = int(round(frames[i])) if np.isfinite(frames[i]) else i
#                             ids.append(f"{obs_id}_{label}_frame{frame_i:06d}")
#                     else:
#                         ids = []
#                         for i in range(npt):
#                             ti = int(point_track[i]) if point_track[i] >= 0 else 0
#                             label = track_labels[ti] if (0 <= ti < len(track_labels)) else "track_unknown"
#                             ids.append(f"{obs_id}_{label}_pt{i:05d}")

#                 tasks.append({
#                     "grp": grp,
#                     "is_sim": is_sim,
#                     "offset": offset,
#                     "xs": xs, "ys": ys, "heights": heights,
#                     "ids": ids,
#                     "point_track": point_track,   # <-- per-point track index
#                     "track_labels": track_labels, # <-- list of labels per track
#                 })

#         # observation-level progress
#         pbar_obs = None
#         if progress_mode == "observation" and show_progress:
#             total_pts = int(sum(t["xs"].size for t in tasks))
#             pbar_obs = tqdm(total=total_pts, desc=f"{obs_id}: viewsheds", unit="pts",
#                             dynamic_ncols=True, leave=False)

#         # Process tasks
#         for t in tasks:
#             grp       = t["grp"]
#             is_sim    = t["is_sim"]
#             offset    = t["offset"]
#             xs        = t["xs"]
#             ys        = t["ys"]
#             heights   = t["heights"]
#             ids       = t["ids"]
#             p2t       = t["point_track"]
#             tr_labels = t["track_labels"]

#             # per-group progress
#             pbar_grp = None
#             if progress_mode == "group" and show_progress:
#                 pbar_grp = tqdm(total=int(xs.size), desc=f"{obs_id} | {grp}", unit="pts",
#                                 dynamic_ncols=True, leave=False)

#             vis = np.full(xs.shape[0], np.nan, dtype=np.float32)

#             for i in range(xs.shape[0]):
#                 X, Y, hgt, step_id = float(xs[i]), float(ys[i]), float(heights[i]), ids[i]

#                 ti = int(p2t[i]) if (i < p2t.size and p2t[i] >= 0) else -1
#                 track_label = tr_labels[ti] if (0 <= ti < len(tr_labels)) else "track_unknown"

#                 folder = _viewshed_folder(
#                     outroot, obs_id,
#                     track_label=track_label,
#                     is_simulated=is_sim,
#                     offset=offset if is_sim else None,
#                     dataset_tag=(auto_dataset_tag if is_sim else None),
#                 )
#                 tif_path = folder / _viewshed_filename(step_id, radius)

#                 # reuse file if allowed and present
#                 reuse = (reuse_simulated if is_sim else reuse_observed)
#                 if reuse and tif_path.exists():
#                     try:
#                         vis[i] = _viewshed_mean_visible(tif_path)
#                     except Exception:
#                         pass  # fall back to compute in-memory

#                 # compute once in memory if needed
#                 if np.isnan(vis[i]):
#                     try:
#                         mem_ds = _viewshed_generate_mem(dsm_path, X, Y, hgt, radius_m=radius)
#                         try:
#                             band = mem_ds.GetRasterBand(1)
#                             stats = band.GetStatistics(0, 1)
#                             vis[i] = float(stats[2]) if stats is not None else np.nan
#                             # persist only if flagged for this class
#                             keep_flag = (keep_simulated if is_sim else keep_observed)
#                             if keep_flag:
#                                 _save_mem_to_gtiff(mem_ds, tif_path, threads=threads)
#                         finally:
#                             mem_ds = None
#                     except Exception as exc:
#                         warnings.warn(f"{obs_id} {grp} point {i}: viewshed error: {exc}")
#                         vis[i] = np.nan

#                 if pbar_obs is not None:
#                     pbar_obs.update(1)
#                 if pbar_grp is not None:
#                     pbar_grp.update(1)

#             if pbar_grp is not None:
#                 pbar_grp.close()

#             out = xr.Dataset({"viewshed_vis": (("point",), vis.astype(np.float32))})
#             out["viewshed_vis"].attrs.update({
#                 "description": f"Proportion visible in {int(radius)} m viewshed (1 visible, 0 invisible).",
#                 "radius_m": float(radius),
#                 "generator": "gdal.ViewshedGenerate (MEM path, optional GTiff persist)",
#                 "visibleVal": 1, "invisibleVal": 0, "nodataVal": -10000,
#             })
#             out.to_zarr(store, group=grp, mode="a")

#         if pbar_obs is not None:
#             pbar_obs.close()

