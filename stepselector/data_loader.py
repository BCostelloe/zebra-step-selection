import pandas as pd
import glob
import os
from torch.utils.data import Dataset
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import warnings


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

    def __init__(
        self,
        target_dir,
        reference_dir,
        social_radius,
        num_ref_steps,
        num_context_steps,
        target_id_col="target_id",
        context_features=None,
        target_features=None,
        use_transformed=True,
        return_dataframe=False,
    ):
        self.context_features = (
            context_features
            if context_features is not None
            else self.DEFAULT_CONTEXT_FEATURES
        )
        self.target_features = (
            target_features
            if target_features is not None
            else self.DEFAULT_TARGET_FEATURES
        )
        self.num_context_steps = num_context_steps
        self.use_transformed = use_transformed
        self.return_dataframe = return_dataframe

        target_files = sorted(glob.glob(os.path.join(target_dir, "*.pkl")))
        target_df = pd.concat(
            (pd.read_pickle(f) for f in target_files), ignore_index=False
        )
        reference_files = sorted(glob.glob(os.path.join(reference_dir, "*.pkl")))
        reference_df = pd.concat(
            (pd.read_pickle(f) for f in reference_files), ignore_index=True
        )

        # Check and warn about NaN values in the target_df
        if target_df.isnull().values.any():
            warnings.warn(
                "NaN values detected in target_df. These rows will be dropped.",
                stacklevel=2,
            )

        # Check and warn about NaN values in the reference_df
        if reference_df.isnull().values.any():
            warnings.warn(
                "NaN values detected in reference_df. These rows will be dropped.",
                stacklevel=2,
            )

        # Remove rows with NaN values
        target_df.dropna(inplace=True)
        reference_df.dropna(inplace=True)

        # One-hot encode ground_class
        onehot_encode_ground = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        ground_combined = pd.concat(
            [target_df[["ground_class"]], reference_df[["ground_class"]]]
        )
        onehot_encode_ground.fit(ground_combined)

        for df in [target_df, reference_df]:
            ground_class_onehot = onehot_encode_ground.transform(df[["ground_class"]])
            new_features = [
                "ground_unclassified",
                "ground_bare",
                "ground_grass",
                "ground_tree",
            ]
            df[new_features] = ground_class_onehot
            df.drop(["ground_class"], axis=1, inplace=True)
            self._update_features("ground_class", new_features)

        # One-hot encode species
        onehot_encode_species = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        species_combined = pd.concat(
            [target_df[["species"]], reference_df[["species"]]]
        )
        onehot_encode_species.fit(species_combined)

        for df in [target_df, reference_df]:
            species_onehot = onehot_encode_species.transform(df[["species"]])
            species_categories = onehot_encode_species.categories_[0]
            new_features = [f"species_{s}" for s in species_categories]
            df[new_features] = species_onehot
            df.drop(["species"], axis=1, inplace=True)
            self._update_features("species", new_features)

        # One-hot encode age_class
        onehot_encode_age = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        age_combined = pd.concat(
            [target_df[["age_class"]], reference_df[["age_class"]]]
        )
        onehot_encode_age.fit(age_combined)

        for df in [target_df, reference_df]:
            age_class_onehot = onehot_encode_age.transform(df[["age_class"]])
            age_classes = onehot_encode_age.categories_[0]
            new_features = [f"age_{a}" for a in age_classes]
            df[new_features] = age_class_onehot
            df.drop(["age_class"], axis=1, inplace=True)
            self._update_features("age_class", new_features)

        # Transformations
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

        # Apply initial transformations (log and angle) to both DataFrames
        for df in [target_df, reference_df]:
            for col in transformations["log"]:
                df[f"{col}_transformed"] = np.log1p(df[col])

            for col in transformations["angle"]:
                df[f"{col}_transformed"] = df[col] / 180

        # Combine DataFrames to calculate Z-score statistics
        combined_df = pd.concat([target_df, reference_df])

        # Apply Z-score normalization
        for df in [target_df, reference_df]:
            for col in transformations["zscore"]:
                transformed_col_name = f"{col}_transformed"

                # If no prior transformation exists, create the transformed version
                if transformed_col_name not in df.columns:
                    df[transformed_col_name] = df[col]

                # Compute Z-score stats from the combined DataFrame
                combined_mean = combined_df[transformed_col_name].mean()
                combined_std = combined_df[transformed_col_name].std()

                # Normalize the transformed version
                df[transformed_col_name] = (
                    df[transformed_col_name] - combined_mean
                ) / combined_std

        target_df["track_id"] = [x.split("f")[0] for x in target_df["id"]]

        self.context_df = target_df.copy()

        keep = [
            target_df[target_df["track_id"] == t].iloc[self.num_context_steps :].copy()
            for t in np.unique(target_df["track_id"])
        ]
        self.new_target_df = pd.concat(keep)

        self.reference_df = reference_df
        self.target_id_col = target_id_col
        self.social_radius = social_radius
        self.num_ref_steps = num_ref_steps
        self.num_samples = len(self.new_target_df)
        self.id_to_ref_indices = self._create_id_to_ref_indices()

    def _update_features(self, original_feature, new_features):
        if original_feature in self.context_features:
            self.context_features.remove(original_feature)
            self.context_features.extend(new_features)
        if original_feature in self.target_features:
            self.target_features.remove(original_feature)
            self.target_features.extend(new_features)

    def _create_id_to_ref_indices(self):
        id_to_ref_indices = {}
        for idx, row in self.reference_df.iterrows():
            target_id = row[self.target_id_col]
            id_to_ref_indices.setdefault(target_id, []).append(idx)
        return id_to_ref_indices

    def __len__(self):
        return len(self.new_target_df)

    def __getitem__(self, idx):
        target_row = self.new_target_df.iloc[idx].copy()
        target_id = target_row[self.target_id_col]
        track_id = target_id.split("f")[0]
        reference_indices = self.id_to_ref_indices.get(target_id, [])
        reference_rows = self.reference_df.iloc[reference_indices].copy()
        reference_rows = reference_rows.sample(self.num_ref_steps)

        context = self.context_df[self.context_df["track_id"] == track_id]
        target_index = self.context_df.index[self.context_df["id"] == target_id][0]
        context_rows = self.context_df.iloc[
            (target_index - self.num_context_steps) : target_index, :
        ].copy()

        # Keep only specified features and ensure numerical type
        selected_context_features = [
            f + "_transformed"
            if self.use_transformed and f"{f}_transformed" in context_rows
            else f
            for f in self.context_features
        ]
        selected_target_features = [
            f + "_transformed"
            if self.use_transformed and f"{f}_transformed" in target_row
            else f
            for f in self.target_features
        ]

        if self.return_dataframe:
            return (
                target_row[selected_target_features],
                reference_rows[selected_target_features],
                context_rows[selected_context_features],
            )

        return (
            target_row[selected_target_features].to_numpy(dtype=np.float32)[None],
            reference_rows[selected_target_features].to_numpy(dtype=np.float32)[
                :, None
            ],
            context_rows[selected_context_features].to_numpy(dtype=np.float32),
        )


def to_dataframe(data):
    """Convert input data (list of Series, DataFrames, or other iterables) to a DataFrame."""
    if isinstance(data[0], pd.Series):
        return pd.concat(data, axis=1).T  # Convert list of Series to DataFrame
    elif isinstance(data[0], pd.DataFrame):
        return pd.concat(data, ignore_index=True)  # Concatenate DataFrames
    else:
        return pd.DataFrame(data)  # Convert other types (lists, tuples) to DataFrame


def collate_df(batch):
    """Collates a batch into DataFrames."""
    targets, references, context = zip(*batch)
    return to_dataframe(targets), to_dataframe(references), to_dataframe(context)
