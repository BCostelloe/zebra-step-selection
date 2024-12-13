import pandas as pd
import glob
import os
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from scipy.special import logit


class ZebraDataset(Dataset):    
    def __init__(self, target_dir, reference_dir, social_radius, num_ref_steps, num_context_steps, target_id_col='target_id', columns_to_keep = None):

        self.columns_to_keep = columns_to_keep if columns_to_keep is not None else []
        self.num_context_steps = 5

        target_files = sorted(glob.glob(os.path.join(target_dir, '*.pkl')))
        target_df = pd.concat((pd.read_pickle(f) for f in target_files), ignore_index = False)
        reference_files = sorted(glob.glob(os.path.join(reference_dir, '*.pkl')))
        reference_df = pd.concat((pd.read_pickle(f) for f in reference_files), ignore_index = True)

        # Initialize encoders
        onehot_encode = OneHotEncoder(sparse_output = False)

        # One-hot encode ground_class
        ground_class_df = target_df[['ground_class']].copy()
        ground_class_onehot = onehot_encode.fit_transform(ground_class_df)
        new_columns = ['ground_unclassified', 'ground_bare', 'ground_grass', 'ground_tree']
        target_df[new_columns] = ground_class_onehot
        target_df.drop(['ground_class'], axis=1, inplace=True)

        ground_class_df = reference_df[['ground_class']].copy()
        ground_class_onehot = onehot_encode.fit_transform(ground_class_df)
        new_columns = ['ground_unclassified', 'ground_bare', 'ground_grass', 'ground_tree']
        reference_df[new_columns] = ground_class_onehot
        reference_df.drop(['ground_class'], axis=1, inplace=True)
        
        # Update columns_to_keep dynamically
        self._update_columns_to_keep('ground_class', new_columns)

        # One-hot encode species
        species_df = target_df[['species']].copy()
        species_onehot = onehot_encode.fit_transform(species_df)
        species_categories = onehot_encode.categories_[0]
        new_columns = [f"species_{s}" for s in species_categories]
        target_df[new_columns] = species_onehot
        target_df.drop(['species'], axis=1, inplace=True)

        species_df = reference_df[['species']].copy()
        species_onehot = onehot_encode.fit_transform(species_df)
        species_categories = onehot_encode.categories_[0]
        new_columns = [f"species_{s}" for s in species_categories]
        reference_df[new_columns] = species_onehot
        reference_df.drop(['species'], axis=1, inplace=True)
        
        # Update columns_to_keep dynamically
        self._update_columns_to_keep('species', new_columns)

        # One-hot encode age_class
        age_class_df = target_df[['age_class']].copy()
        age_class_onehot = onehot_encode.fit_transform(age_class_df)
        age_classes = onehot_encode.categories_[0]
        new_columns = [f"age_{a}" for a in age_classes]
        target_df[new_columns] = age_class_onehot
        target_df.drop(['age_class'], axis=1, inplace=True)

        age_class_df = reference_df[['age_class']].copy()
        age_class_onehot = onehot_encode.fit_transform(age_class_df)
        age_classes = onehot_encode.categories_[0]
        new_columns = [f"age_{a}" for a in age_classes]
        reference_df[new_columns] = age_class_onehot
        reference_df.drop(['age_class'], axis=1, inplace=True)
        
        # Update columns_to_keep dynamically
        self._update_columns_to_keep('age_class', new_columns)

        # Transform & scale data
        cols_to_logtransform = ['step_speed_mps', 'dist_to_observer']
        angle_cols = ['angle_to_observers']
        cols_to_logittransform = ['angle_to_observers', 'viewshed_vis']
        cols_to_zscore = ['step_speed_mps', 'angle_to_observers', 'dist_to_observer', 'delta_observer_dist', 'ground_slope', 'viewshed_vis']

        for c in cols_to_logtransform:
            target_df[c] = np.log1p(target_df[c])
            reference_df[c] = np.log1p(reference_df[c])

        for c in angle_cols:
            target_df[c] = target_df[c]/180
            reference_df[c] = reference_df[c]/180

        for c in cols_to_logittransform:
            target_df[c] = logit(target_df[c])
            reference_df[c] = logit(reference_df[c])

        for c in cols_to_zscore:
            mean_val = np.mean(target_df[c])
            std_val = np.std(target_df[c])
            target_df[c] = (target_df[c] - mean_val)/std_val
            reference_df[c] = (reference_df[c] - mean_val)/std_val

        
        keep = []
        tracks = np.unique([x.split('f')[0] for x in target_df['id']])
        target_df['track_id'] = [x.split('f')[0] for x in target_df['id']]
        self.context_df = target_df.copy() # full target dataset is available to use as context
        for t in tracks:
            sub_data = target_df[target_df['track_id'] == t].iloc[self.num_context_steps:].copy()
            keep.append(sub_data)
        new_target_df = pd.concat(keep)
        
        self.new_target_df = new_target_df
        self.reference_df = reference_df
        self.target_id_col = target_id_col
        self.social_radius = social_radius
        self.num_ref_steps = num_ref_steps
        self.num_samples = len(self.new_target_df)

        # Create mapping of target ID to reference indices
        self.id_to_ref_indices = self._create_id_to_ref_indices()

    def _update_columns_to_keep(self, original_column, new_columns):
        """
        Update columns_to_keep by replacing an original column with new columns.
    
        Parameters:
            original_column (str): The name of the original column being replaced.
            new_columns (list): List of new columns to add in place of the original column.
        """
        if original_column in self.columns_to_keep:
            self.columns_to_keep.remove(original_column)
            self.columns_to_keep.extend(new_columns)

    def _create_id_to_ref_indices(self):
        id_to_ref_indices = {}
        for idx, row in self.reference_df.iterrows():
            target_id = row[self.target_id_col]
            if target_id not in id_to_ref_indices:
                id_to_ref_indices[target_id] = []
            id_to_ref_indices[target_id].append(idx)
        return id_to_ref_indices
            
    def __len__(self):
        return len(self.new_target_df)

    def __getitem__(self, idx):
        target_row = self.new_target_df.iloc[idx].copy()
        target_id = target_row[self.target_id_col]
        track_id = target_id.split('f')[0]
        reference_indices = self.id_to_ref_indices.get(target_id, [])
        reference_rows = self.reference_df.iloc[reference_indices].copy()
        
        # # Debugging: Log the target ID and the number of available reference rows
        # print(f"Processing target ID {target_id} with {len(reference_rows)} reference rows available.")
        
        reference_rows = reference_rows.sample(self.num_ref_steps)
        observation_name = 'observation' + target_id.split('_')[0].split('b')[1]
        reference_rows['observation'] = observation_name
        context = self.context_df[self.context_df['track_id'] == track_id]
        target_index = self.context_df.index[self.context_df['id'] == target_id][0]
        context_rows = self.context_df.iloc[(target_index-self.num_context_steps):target_index,:].copy()
        
        # keep only specified columns and convert to dictionary
        if self.columns_to_keep:
            target_data = target_row[self.columns_to_keep].to_dict()
            reference_data = reference_rows[self.columns_to_keep].to_dict(orient='records')
            context_data = context_rows[self.columns_to_keep].to_dict(orient='records')
        else:
            target_data = target_row.to_dict()
            reference_data = reference_rows.to_dict(orient = 'records')
            context_data = context_rows.to_dict(orient = 'records')

        return target_data, reference_data, context_data

def custom_collate(batch):
        targets, references, context = zip(*batch)
        return targets, references, context
