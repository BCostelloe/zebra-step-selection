import glob
import os
import warnings
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, TensorDataset
from sklearn.preprocessing import OneHotEncoder
from joblib import delayed, Parallel
import torch
import tqdm

class ZebraDataset(Dataset):
    DEFAULT_CONTEXT_FEATURES = [
        "target_id",
        "observation",
        "step_speed_mps",
        "angle_to_observers",
        "dist_to_observer",
        "delta_observer_dist",
        "road",
        "ground_class",
        "ground_bare",
        "ground_tree",
        "ground_slope",
        "viewshed_vis",
        "social_dens",
        "social_vis",
        "age_class",
        "species",
        "individual_ID",
    ]
    DEFAULT_TARGET_FEATURES = [
        "target_id",
        "observation",
        "step_speed_mps",
        "angle_to_observers",
        "dist_to_observer",
        "delta_observer_dist",
        "road",
        "ground_bare",
        "ground_tree",
        "ground_slope",
        "viewshed_vis",
        "social_dens",
        "social_vis",
        "age_class",
        "species",
        "individual_ID",
    ]

    def __init__(
        self,
        target_dir,
        reference_dir,
        num_ref_steps,
        num_context_steps,
        target_id_col="target_id",
        context_features=None,
        target_features=None,
        use_transformed=True,
        return_dataframe=False,
        random_effects=None,
    ):
        self.context_features = (
            context_features
            if context_features is not None
            else self.DEFAULT_CONTEXT_FEATURES.copy()
        )
        self.target_features = (
            target_features
            if target_features is not None
            else self.DEFAULT_TARGET_FEATURES.copy()
        )
        self.num_context_steps = num_context_steps
        self.use_transformed = use_transformed
        self.return_dataframe = return_dataframe
        self.random_effects = random_effects  # may be None
        self.target_id_col = target_id_col
        self.num_ref_steps = num_ref_steps

        # --- Load data in parallel using Pandas ---
        target_files = sorted(glob.glob(os.path.join(target_dir, "*.pkl")))
        reference_files = sorted(glob.glob(os.path.join(reference_dir, "*.pkl")))
        target_df = pd.concat(
            Parallel(n_jobs=-1)(delayed(pd.read_pickle)(f) for f in target_files),
            ignore_index=True,
        )
        reference_df = pd.concat(
            Parallel(n_jobs=-1)(delayed(pd.read_pickle)(f) for f in reference_files),
            ignore_index=True,
        )

        # Warn and drop NaNs if any
        if target_df.isnull().values.any():
            warnings.warn(
                "NaN values detected in target_df. These rows will be dropped.",
                stacklevel=2,
            )
        if reference_df.isnull().values.any():
            warnings.warn(
                "NaN values detected in reference_df. These rows will be dropped.",
                stacklevel=2,
            )
        target_df.dropna(inplace=True)
        reference_df.dropna(inplace=True)

        # --- One-hot encoding ---
        # ground_class
        onehot_ground = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        ground_combined = pd.concat(
            [target_df[["ground_class"]], reference_df[["ground_class"]]]
        )
        onehot_ground.fit(ground_combined)
        new_ground_features = [
            "ground_class_unclassified",
            "ground_class_bare",
            "ground_class_grass",
            "ground_class_tree",
        ]
        for df in [target_df, reference_df]:
            onehot = onehot_ground.transform(df[["ground_class"]])
            df[new_ground_features] = onehot
            df.drop(["ground_class"], axis=1, inplace=True)
            self._update_features("ground_class", new_ground_features)

        # species
        onehot_species = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        species_combined = pd.concat(
            [target_df[["species"]], reference_df[["species"]]]
        )
        onehot_species.fit(species_combined)
        species_categories = onehot_species.categories_[0]
        new_species_features = [f"species_{s}" for s in species_categories]
        for df in [target_df, reference_df]:
            onehot = onehot_species.transform(df[["species"]])
            df[new_species_features] = onehot
            df.drop(["species"], axis=1, inplace=True)
            self._update_features("species", new_species_features)

        # age_class
        onehot_age = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        age_combined = pd.concat(
            [target_df[["age_class"]], reference_df[["age_class"]]]
        )
        onehot_age.fit(age_combined)
        age_categories = onehot_age.categories_[0]
        new_age_features = [f"age_{a}" for a in age_categories]
        for df in [target_df, reference_df]:
            onehot = onehot_age.transform(df[["age_class"]])
            df[new_age_features] = onehot
            df.drop(["age_class"], axis=1, inplace=True)
            self._update_features("age_class", new_age_features)

        # --- Transformations ---
        transformations = {
            "log": ["step_speed_mps", "dist_to_observer", "social_dens", "social_vis"],
            "angle": ["angle_to_observers"],
            "zscore": [
                "step_speed_mps",
                "angle_to_observers",
                "dist_to_observer",
                "delta_observer_dist",
                "ground_slope",
                "viewshed_vis",
                "social_dens",
                "social_vis",
            ],
        }
        for df in [target_df, reference_df]:
            for col in transformations["log"]:
                df[f"{col}_transformed"] = np.log1p(df[col])
            for col in transformations["angle"]:
                df[f"{col}_transformed"] = df[col] / 180
        for df in [target_df, reference_df]:
            for col in transformations["zscore"]:
                tcol = f"{col}_transformed"
                if tcol not in df.columns:
                    df[tcol] = df[col]
        combined_df = pd.concat([target_df, reference_df], axis=0)
        for df in [target_df, reference_df]:
            for col in transformations["zscore"]:
                tcol = f"{col}_transformed"
                combined_mean = combined_df[tcol].mean()
                combined_std = combined_df[tcol].std()
                df[tcol] = (df[tcol] - combined_mean) / combined_std

        # --- Modified Indexing Section ---
        # Compute track identifier from the "id" column (assumes "id" exists)
        target_df["track_id"] = target_df["id"].apply(lambda x: x.split("f")[0])
        target_df["local_index"] = target_df.groupby("track_id").cumcount()

        # One-hot encoding for track_id
        onehot_track = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        onehot_track.fit(target_df[["track_id"]])
        track_id_onehot = onehot_track.transform(target_df[["track_id"]])
        track_categories = onehot_track.categories_[0]
        new_track_features = [f"track_{t}" for t in track_categories]
        track_df = pd.DataFrame(
            track_id_onehot, columns=new_track_features, index=target_df.index
        )
        target_df = pd.concat([target_df, track_df], axis=1)
        self._update_features("track_id", new_track_features)

        # Valid targets: those with at least num_context_steps preceding rows.
        self.valid_indices = target_df[
            target_df["local_index"] >= self.num_context_steps
        ].index.tolist()
        self.num_samples = len(self.valid_indices)

        # Build mapping from target_id to reference indices (precomputed)
        self.id_to_ref_indices = {}
        for idx, row in reference_df.iterrows():
            tid = row[self.target_id_col]
            self.id_to_ref_indices.setdefault(tid, []).append(idx)

        # --- Precompute track to context mapping ---
        # For each track, store a sorted list of (local_index, global index)
        self.track_to_context = {}
        for track, group in target_df.groupby("track_id"):
            # Group should already be in order.
            self.track_to_context[track] = list(
                zip(group["local_index"].values, group.index.values)
            )

        self.target_df = target_df
        self.reference_df = reference_df

    def _update_features(self, original_feature, new_features):
        if original_feature in self.context_features:
            self.context_features.remove(original_feature)
            self.context_features.extend(new_features)
        if original_feature in self.target_features:
            self.target_features.remove(original_feature)
            self.target_features.extend(new_features)
        if self.random_effects:
            if original_feature in self.random_effects:
                self.random_effects.remove(original_feature)
                self.random_effects.extend(new_features)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Retrieve target row by valid index.
        global_idx = self.valid_indices[idx]
        target_row = self.target_df.loc[global_idx].copy()
        target_id = target_row[self.target_id_col]
        track_id = target_row["track_id"]
        L = int(target_row["local_index"])

        # --- Get reference rows ---
        ref_indices = self.id_to_ref_indices.get(target_id, [])
        if len(ref_indices) < self.num_ref_steps:
            raise RuntimeError(f"Not enough reference rows for target_id {target_id}")
        ref_sample = np.random.choice(
            ref_indices, size=self.num_ref_steps, replace=False
        )
        ref_rows = self.reference_df.loc[ref_sample].copy()

        # --- Get context rows ---
        context_candidates = [
            idx for li, idx in self.track_to_context[track_id] if li < L
        ]
        if len(context_candidates) >= self.num_context_steps:
            context_indices = context_candidates[-self.num_context_steps :]
        else:
            context_indices = context_candidates
        context_rows = self.target_df.loc[context_indices].copy()

        # --- Select features ---
        selected_context_features = [
            f + "_transformed"
            if self.use_transformed and f"{f}_transformed" in context_rows.columns
            else f
            for f in self.context_features
        ]
        selected_target_features = [
            f + "_transformed"
            if self.use_transformed and f"{f}_transformed" in target_row.index
            else f
            for f in self.target_features
        ]
        selected_random_effects = (
            [
                f + "_transformed"
                if self.use_transformed and f"{f}_transformed" in target_row.index
                else f
                for f in self.random_effects
            ]
            if self.random_effects
            else []
        )

        # --- For numpy conversion, filter out non-numeric columns ---
        if not self.return_dataframe:
            selected_context_features = [
                col
                for col in selected_context_features
                if pd.api.types.is_numeric_dtype(self.target_df[col].dtype)
            ]
            selected_target_features = [
                col
                for col in selected_target_features
                if pd.api.types.is_numeric_dtype(self.target_df[col].dtype)
            ]
            if selected_random_effects:
                selected_random_effects = [
                    col
                    for col in selected_random_effects
                    if pd.api.types.is_numeric_dtype(self.target_df[col].dtype)
                ]

        # --- Pad context rows with zeros if needed ---
        current_context_count = context_rows[selected_context_features].shape[0]
        if current_context_count < self.num_context_steps:
            missing = self.num_context_steps - current_context_count
            pad_df = pd.DataFrame(
                np.zeros((missing, len(selected_context_features)), dtype=np.float32),
                columns=selected_context_features,
            )
            if context_rows.empty:
                context_rows = pad_df
            else:
                # Ensure we only take the numeric columns before concatenation.
                context_rows = pd.concat(
                    [context_rows[selected_context_features], pad_df],
                    ignore_index=True,
                )
        else:
            context_rows = context_rows[selected_context_features]

        if self.return_dataframe:
            return (
                target_row[selected_target_features],
                ref_rows[selected_target_features],
                context_rows,
                target_row[selected_random_effects]
                if selected_random_effects
                else None,
            )
        else:
            return (
                target_row[selected_target_features].to_numpy(dtype=np.float32)[None],
                ref_rows[selected_target_features].to_numpy(dtype=np.float32)[:, None],
                context_rows.to_numpy(dtype=np.float32),
                target_row[selected_random_effects].to_numpy(dtype=np.float32)[None]
                if selected_random_effects
                else None,
            )


def cache_dataset(original_dataloader, num_cache_passes=2):
    """
    Iterates through the original_dataloader multiple times, caching the batches.
    Assumes each batch is a tuple: (target, reference, context, random_effects).
    Returns a TensorDataset containing the cached data.
    """
    all_target, all_reference, all_context, all_random_effects = [], [], [], []
    for _ in range(num_cache_passes):
        for batch in tqdm(original_dataloader, desc="Caching dataset"):
            target, reference, context, random_effects = batch
            all_target.append(target)
            all_reference.append(reference)
            all_context.append(context)
            all_random_effects.append(random_effects)
    cached_target = torch.cat(all_target, dim=0)
    cached_reference = torch.cat(all_reference, dim=0)
    cached_context = torch.cat(all_context, dim=0)
    cached_random_effects = torch.cat(all_random_effects, dim=0)
    return TensorDataset(
        cached_target, cached_reference, cached_context, cached_random_effects
    )


def to_dataframe(data):
    """Convert a list of Series, DataFrames, or other iterables to a DataFrame."""
    if isinstance(data[0], pd.Series):
        return pd.concat(data, axis=1).T
    elif isinstance(data[0], pd.DataFrame):
        return pd.concat(data, ignore_index=True)
    else:
        return pd.DataFrame(data)


def collate_df(batch):
    """Collates a batch into DataFrames."""
    targets, references, context, *rest = zip(*batch)
    return to_dataframe(targets), to_dataframe(references), to_dataframe(context)
