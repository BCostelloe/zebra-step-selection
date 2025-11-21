import math
from pathlib import Path
import numpy as np
import xarray as xr
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Optional: if you want PyTorch integration
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class Dataset:
        pass  # dummy base class

########
# FEATURE AND TRANSFORM CONFIGURATION
########

NUMERIC_FEATURES = [
    "angle_to_observer",
    "delta_observer_dist",
    "dist_to_observer",
    "step_speed_mps",
    "ground_slope_pct",
    "ground_slope_deg",
    "nn_distance_m",
    "trail_probability",
    "viewshed_vis",
    "social_density",
    "social_visible_count",
    "viewshed_vis",
]

CATEGORICAL_FEATURES = [
    "age",
    "nn_age",
    "species",
    "nn_species",
    "ground_class",
    "ground_class_name",
    "nn_visible",   # treated as categorical {0,1}
]

# Helper to avoid log / logit problems
LOG_EPS = 1e-6


@dataclass
class NumericStats:
    mean: float
    std: float

@dataclass
class GlobalStats:
    # maps feature name -> NumericStats (already *after* whatever transform: log/logit/angle-rescale)
    numeric_stats: Dict[str, NumericStats]
    # maps feature name -> {category -> index}
    category_maps: Dict[str, Dict[Any, int]]


#########
# COMPUTE GLOBAL STATS FROM OBSERVED STEPS
#########

def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, LOG_EPS, 1.0 - LOG_EPS)
    return np.log(p / (1.0 - p))

def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.clip(x, LOG_EPS, np.inf))

def _transform_numeric_raw(name: str, arr: np.ndarray) -> np.ndarray:
    """Apply the *pre-zscore* transform for a single numeric feature."""
    if name == "angle_to_observer":
        return arr / 180.0
    elif name in ("trail_probability", "viewshed_vis"):
        return _logit(arr)
    elif name in ("dist_to_observer", "step_speed_mps"):
        return _safe_log(arr)
    # z-score only:
    elif name in ("delta_observer_dist", "nn_distance_m",
                  "ground_slope_deg", "ground_slope_pct",
                  "social_density", "social_visible_count"):
        return arr
    else:
        # default: identity
        return arr


def compute_global_stats_for_observed(
    zarr_root: str | Path,
    step_length: float,
    offsets_forward: List[float],
    offsets_backward: List[float],
    *,
    obs_to_process: str | List[str] | None = None,
    directions: Tuple[str, ...] = ("forward", "backward"),
) -> GlobalStats:
    """
    Scan all observed groups for a given step_length and offsets,
    compute:
      - mean & std of transformed numeric features
      - category vocabularies for categorical features

    This uses only OBSERVED groups; simulations are transformed with the same stats.
    """
    zroot = Path(zarr_root)
    stores = _discover_stores(zroot, obs_to_process)

    # Welford accumulators: feature -> (n, mean, M2)
    running = {f: [0, 0.0, 0.0] for f in NUMERIC_FEATURES}

    # categorical vocabularies: feature -> set of values
    cat_values: Dict[str, set] = {f: set() for f in CATEGORICAL_FEATURES}

    def _update_running(name: str, vals: np.ndarray):
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return
        n_old, mean_old, M2_old = running[name]
        # Welford's algorithm
        for x in vals:
            n_old += 1
            delta = x - mean_old
            mean_old += delta / n_old
            M2_old += delta * (x - mean_old)
        running[name] = [n_old, mean_old, M2_old]

    for store in stores:
        root = xr.open_zarr(store)
        obs_id = root.attrs.get("observation_id", store.stem)

        for direction in directions:
            if direction == "forward":
                offsets = offsets_forward
            elif direction == "backward":
                offsets = offsets_backward
            else:
                continue

            for offset in offsets:
                grp = _group_observed_dir(step_length, offset, direction)
                try:
                    ds = xr.open_zarr(store, group=grp)
                except Exception:
                    continue

                # numeric features
                for fname in NUMERIC_FEATURES:
                    if fname not in ds:
                        continue
                    arr = ds[fname].values.astype(np.float64)
                    arr_t = _transform_numeric_raw(fname, arr)
                    _update_running(fname, arr_t)

                # categorical features
                for cf in CATEGORICAL_FEATURES:
                    if cf == "age":
                        # from track-level coord, if present
                        if "age" in ds.coords:
                            vals = np.asarray(ds["age"].values).astype(object).ravel()
                            for v in vals:
                                if v is not None:
                                    cat_values["age"].add(str(v))
                    elif cf in ("species", "individual_ID"):
                        # these are also likely in coords
                        if cf in ds.coords:
                            vals = np.asarray(ds[cf].values).astype(object).ravel()
                            for v in vals:
                                if v is not None:
                                    cat_values[cf].add(str(v))
                    else:
                        # point-level categorical features
                        if cf in ds:
                            vals = np.asarray(ds[cf].values).astype(object).ravel()
                            for v in vals:
                                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                                    cat_values[cf].add(str(v))

    # finalize numeric stats
    numeric_stats: Dict[str, NumericStats] = {}
    for name, (n, mean, M2) in running.items():
        if n < 2:
            numeric_stats[name] = NumericStats(mean=0.0, std=1.0)
        else:
            var = M2 / (n - 1)
            std = math.sqrt(var) if var > 0 else 1.0
            numeric_stats[name] = NumericStats(mean=mean, std=std)

    # finalize category maps: sorted for reproducibility
    category_maps: Dict[str, Dict[Any, int]] = {}
    for name, s in cat_values.items():
        cats = sorted(list(s))
        category_maps[name] = {c: i for i, c in enumerate(cats)}

    return GlobalStats(numeric_stats=numeric_stats, category_maps=category_maps)


#############
# DATASET BUILDER
#############

@dataclass
class StepIndexRow:
    store_idx: int
    direction: str
    offset: float
    track_idx: int
    obs_point_idx: int       # global point index in observed group
    endpoint_local_idx: int  # index within track block (0..c-1); endpoint j_end
    sim_start: int           # start index of sim block within sim group
    sim_count: int           # number of sim endpoints for this step

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

def _group_observed_dir(L, O, direction: str) -> str:
    """
    direction: 'forward' or 'backward'
    """
    if direction == "forward":
        return f"steps_{int(L)}m/offset_{int(O)}m/observed"
    elif direction == "backward":
        return f"steps_{int(L)}m/offset_{int(O)}m/observed_backward"
    else:
        raise ValueError(f"Unknown direction: {direction!r}")


def _group_simulated_dir(L, O, direction: str) -> str:
    """
    direction: 'forward' or 'backward'
    """
    if direction == "forward":
        return f"steps_{int(L)}m/offset_{int(O)}m/simulated"
    elif direction == "backward":
        return f"steps_{int(L)}m/offset_{int(O)}m/simulated_backward"
    else:
        raise ValueError(f"Unknown direction: {direction!r}")

class StepSelectionDataset(Dataset):
    """
    Each item:
      - observed_features: (F_obs,)
      - context_features: (N, F_ctx)
      - simulated_features: (S, F_sim)
      - metadata: dict (ids, offsets, direction, etc.)
    """
    def __init__(
        self,
        zarr_root: str | Path,
        step_length: float,
        offsets_forward: List[float],
        offsets_backward: List[float],
        global_stats: GlobalStats,
        *,
        S: int = 10,   # number of simulated steps per observed
        N: int = 5,    # number of context steps
        obs_to_process: str | List[str] | None = None,
        directions: Tuple[str, ...] = ("forward", "backward"),
        features_observed: Optional[List[str]] = None,
        features_simulated: Optional[List[str]] = None,
        features_context: Optional[List[str]] = None,
        max_sim_per_step: int = 40,
        seed: int = 123,
    ):
        super().__init__()
        self.zroot = Path(zarr_root)
        self.step_length = float(step_length)
        self.offsets_forward = offsets_forward
        self.offsets_backward = offsets_backward
        self.global_stats = global_stats
        self.S = S
        self.N = N
        self.directions = directions
        self.max_sim_per_step = max_sim_per_step
        self.rng = random.Random(seed)

        if features_observed is None:
            features_observed = [
                "angle_to_observer",
                "delta_observer_dist",
                "dist_to_observer",
                "step_speed_mps",
                "ground_slope_pct",
                "ground_slope_deg",
                "ground_class",
                "trail_probability",
                "viewshed_vis",
                "social_density",
                "social_visible_count",
                "nn_distance_m",
                "nn_visible",
                "nn_individual_ID",
                "nn_species",
                "age",
                "individual_ID",
                "species",
            ]
        if features_simulated is None:
            # same set minus some observed-only pieces if you like;
            # for now use same list (missing vars handled gracefully)
            features_simulated = list(features_observed)
        if features_context is None:
            # usually same feature set as observed
            features_context = list(features_observed)

        self.features_observed = features_observed
        self.features_simulated = features_simulated
        self.features_context = features_context

        # discover stores
        self.stores = _discover_stores(self.zroot, obs_to_process)

        # keep open datasets per (store_idx, direction, offset)
        self.obs_dsets: Dict[Tuple[int, str, float], xr.Dataset] = {}
        self.sim_dsets: Dict[Tuple[int, str, float], xr.Dataset] = {}

        # build step index
        self.index: List[StepIndexRow] = []
        self._build_index()

    # ---------- internal helpers ----------

    def _get_obs_ds(self, store_idx: int, direction: str, offset: float) -> xr.Dataset:
        key = (store_idx, direction, float(offset))
        if key not in self.obs_dsets:
            store = self.stores[store_idx]
            grp = _group_observed_dir(self.step_length, offset, direction)
            self.obs_dsets[key] = xr.open_zarr(store, group=grp)
        return self.obs_dsets[key]

    def _get_sim_ds(self, store_idx: int, direction: str, offset: float) -> xr.Dataset:
        key = (store_idx, direction, float(offset))
        if key not in self.sim_dsets:
            store = self.stores[store_idx]
            grp = _group_simulated_dir(self.step_length, offset, direction)
            self.sim_dsets[key] = xr.open_zarr(store, group=grp)
        return self.sim_dsets[key]

    def _build_index(self):
        """
        For each observed group:
          - for each track, skip first N endpoints
          - for each endpoint j_end >= N, find simulated block for step j_start = j_end - 1
          - if sim_count >= 1, create index row
        """
        for si, store in enumerate(self.stores):
            root = xr.open_zarr(store)
            obs_id = root.attrs.get("observation_id", store.stem)

            for direction in self.directions:
                if direction == "forward":
                    offsets = self.offsets_forward
                elif direction == "backward":
                    offsets = self.offsets_backward
                else:
                    continue

                for offset in offsets:
                    # observed group
                    try:
                        obs_ds = self._get_obs_ds(si, direction, offset)
                    except Exception:
                        continue
                    if ("track_start" not in obs_ds) or ("track_count" not in obs_ds):
                        continue
                    starts = obs_ds["track_start"].values.astype(np.int64)
                    counts = obs_ds["track_count"].values.astype(np.int64)

                    # simulated group
                    try:
                        sim_ds = self._get_sim_ds(si, direction, offset)
                    except Exception:
                        continue
                    if ("track_start" not in sim_ds or
                        "track_count" not in sim_ds or
                        "target_step_index" not in sim_ds):
                        continue
                    st_sim = sim_ds["track_start"].values.astype(np.int64)
                    ct_sim = sim_ds["track_count"].values.astype(np.int64)
                    tsi = sim_ds["target_step_index"].values.astype(np.int64)

                    n_tracks = starts.size
                    for ti in range(n_tracks):
                        s_obs = starts[ti]
                        c_obs = counts[ti]
                        if c_obs <= self.N:
                            continue  # not enough context

                        sS = st_sim[ti]
                        cS = ct_sim[ti]
                        if cS == 0:
                            continue

                        tsi_block = tsi[sS:sS + cS]   # per-sim step index within this track

                        # endpoints in this track: local indices 0..c_obs-1
                        # step j_start is between local j_start and j_start+1
                        # endpoint index j_end = j_start+1 -> local ≥ 1
                        for endpoint_local in range(self.N, c_obs):  # skip first N
                            j_end = endpoint_local
                            j_start = j_end - 1  # step index we simulated against

                            # find sim rows with target_step_index == j_start
                            mask = (tsi_block == j_start)
                            if not np.any(mask):
                                continue
                            idxs = np.nonzero(mask)[0]
                            sim_start = sS + idxs[0]
                            sim_count = int(idxs.size)
                            if sim_count <= 0:
                                continue

                            p_obs = s_obs + j_end
                            self.index.append(
                                StepIndexRow(
                                    store_idx=si,
                                    direction=direction,
                                    offset=float(offset),
                                    track_idx=ti,
                                    obs_point_idx=int(p_obs),
                                    endpoint_local_idx=j_end,
                                    sim_start=int(sim_start),
                                    sim_count=int(sim_count),
                                )
                            )

        print(f"Built step index with {len(self.index)} samples.")

    # ---------- transforms & encoders ----------

    def _encode_categorical(self, name: str, value: Any) -> np.ndarray:
        cmap = self.global_stats.category_maps.get(name, None)
        if cmap is None or value is None or (isinstance(value, float) and math.isnan(value)):
            return np.zeros((0,), dtype=np.float32)  # no encoding
        v = str(value)
        n_cat = len(cmap)
        out = np.zeros((n_cat,), dtype=np.float32)
        if v in cmap:
            out[cmap[v]] = 1.0
        return out

    def _transform_numeric(self, name: str, x: np.ndarray) -> np.ndarray:
        """
        x is 1D (num_points,) or scalar-like.
        Apply pre-transform then z-score using global stats.
        """
        x = np.asarray(x, dtype=np.float64)
        x_t = _transform_numeric_raw(name, x)
        stats = self.global_stats.numeric_stats.get(name, NumericStats(mean=0.0, std=1.0))
        return (x_t - stats.mean) / (stats.std if stats.std > 0 else 1.0)

    def _extract_feature_vector(
        self,
        ds: xr.Dataset,
        point_idx: int,
        feature_names: List[str],
        track_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Pulls all requested features for a single point (and optional track-level metadata),
        applies transforms & one-hot encodings, and concatenates into a 1D float32 vector.
        """
        pieces = []

        # track-level coords (age, individual_ID, species, etc.)
        if track_idx is not None:
            # age
            if "age" in ds.coords and "track" in ds["age"].dims:
                age_val = str(ds["age"].values[track_idx])
            else:
                age_val = None
            if "individual_ID" in ds.coords and "track" in ds["individual_ID"].dims:
                indiv_val = str(ds["individual_ID"].values[track_idx])
            else:
                indiv_val = None
            if "species" in ds.coords and "track" in ds["species"].dims:
                species_val = str(ds["species"].values[track_idx])
            else:
                species_val = None
        else:
            age_val = indiv_val = species_val = None

        for name in feature_names:
            if name in CATEGORICAL_FEATURES:
                # handle special track-level cases
                if name == "age":
                    enc = self._encode_categorical("age", age_val)
                elif name == "species":
                    enc = self._encode_categorical("species", species_val)
                elif name == "individual_ID":
                    enc = self._encode_categorical("individual_ID", indiv_val)
                else:
                    # point-level categorical
                    if name in ds:
                        v = ds[name].values[point_idx]
                    else:
                        v = None
                    enc = self._encode_categorical(name, v)
                pieces.append(enc.astype(np.float32))
            elif name in NUMERIC_FEATURES:
                if name in ds:
                    v = ds[name].values[point_idx]
                    if np.isnan(v):
                        # keep NaNs as zero after transform (or you can impute differently)
                        v_arr = np.array([np.nan], dtype=np.float64)
                    else:
                        v_arr = np.array([v], dtype=np.float64)
                    v_t = self._transform_numeric(name, v_arr)
                    pieces.append(v_t.astype(np.float32))
                else:
                    pieces.append(np.zeros((1,), dtype=np.float32))
            elif name in ("nn_individual_ID", "nn_species", "nn_age"):
                # these are stored as point-level strings
                if name in ds:
                    v = ds[name].values[point_idx]
                else:
                    v = None
                enc = self._encode_categorical(name.replace("nn_", ""), v)
                pieces.append(enc.astype(np.float32))
            elif name in ("observation_ID", "offset"):
                # metadata handled separately; skip here
                continue
            else:
                # unknown feature: skip
                continue

        if pieces:
            return np.concatenate(pieces, axis=0)
        else:
            return np.zeros((0,), dtype=np.float32)

    # ---------- Dataset interface ----------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        row = self.index[idx]
        store = self.stores[row.store_idx]
        obs_id = xr.open_zarr(store).attrs.get("observation_id", store.stem)

        obs_ds = self._get_obs_ds(row.store_idx, row.direction, row.offset)
        sim_ds = self._get_sim_ds(row.store_idx, row.direction, row.offset)

        # ---------- observed features (at endpoint) ----------
        obs_feat = self._extract_feature_vector(
            obs_ds,
            point_idx=row.obs_point_idx,
            feature_names=self.features_observed,
            track_idx=row.track_idx,
        )

        # ---------- context features ----------
        # N previous endpoints in same track
        s_obs = obs_ds["track_start"].values[row.track_idx]
        # local endpoint index: row.endpoint_local_idx
        ctx_feats = []
        for k in range(1, self.N + 1):
            pt_local = row.endpoint_local_idx - k
            pt_global = s_obs + pt_local
            ctx_feat_k = self._extract_feature_vector(
                obs_ds,
                point_idx=int(pt_global),
                feature_names=self.features_context,
                track_idx=row.track_idx,
            )
            ctx_feats.append(ctx_feat_k)
        # reverse to have oldest→newest or keep as most recent first; here: oldest first
        ctx_feats = ctx_feats[::-1]
        ctx_feats_arr = np.stack(ctx_feats, axis=0)  # (N, F_ctx)

        # ---------- simulated features ----------
        sim_start = row.sim_start
        sim_count = row.sim_count
        # choose indices for S replicates
        if sim_count <= self.S:
            chosen = np.arange(sim_start, sim_start + sim_count, dtype=np.int64)
        else:
            # sample without replacement
            chosen_local = np.array(
                self.rng.sample(range(sim_count), self.S),
                dtype=np.int64,
            )
            chosen = sim_start + chosen_local

        sim_feats = []
        for p in chosen:
            sim_feat_p = self._extract_feature_vector(
                sim_ds,
                point_idx=int(p),
                feature_names=self.features_simulated,
                track_idx=row.track_idx,
            )
            sim_feats.append(sim_feat_p)
        sim_feats_arr = np.stack(sim_feats, axis=0)  # (S, F_sim)

        # ---------- metadata ----------
        meta = {
            "observation_ID": obs_id,
            "offset": row.offset,
            "direction": row.direction,
            "track_idx": row.track_idx,
            "obs_point_idx": row.obs_point_idx,
            "endpoint_local_idx": row.endpoint_local_idx,
            "step_length_m": self.step_length,
        }

        if TORCH_AVAILABLE:
            obs_tensor = torch.from_numpy(obs_feat)
            ctx_tensor = torch.from_numpy(ctx_feats_arr)
            sim_tensor = torch.from_numpy(sim_feats_arr)
            return {
                "observed": obs_tensor,
                "context": ctx_tensor,
                "simulated": sim_tensor,
                "meta": meta,
            }
        else:
            return {
                "observed": obs_feat,
                "context": ctx_feats_arr,
                "simulated": sim_feats_arr,
                "meta": meta,
            }


