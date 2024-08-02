import pandas as pd
import glob
import os
from stepselector.viewshed import generate_viewshed, generate_downsample_viewshed
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np

class ZebraDataset(Dataset):
    def __init__(self, target_dir, reference_dir, rasters_dir, ob_metadata_file, viewshed_radius, viewshed_hw, threads, social_radius, num_ref_steps, target_id_col='target_id', columns_to_keep = None):
        target_files = glob.glob(os.path.join(target_dir, '*.pkl'))
        target_df = pd.concat((pd.read_pickle(f) for f in target_files), ignore_index = True)
        target_df = target_df[target_df.prev_step.str.contains('_', na= False)]
        #df[~df.C.str.contains("XYZ")]
        reference_files = glob.glob(os.path.join(reference_dir, '*.pkl'))
        reference_df = pd.concat((pd.read_pickle(f) for f in reference_files), ignore_index = True)
        metadata_df = pd.read_csv(ob_metadata_file)
        self.metadata_df = metadata_df
        self.target_df = target_df
        self.reference_df = reference_df
        self.target_id_col = target_id_col
        self.columns_to_keep = columns_to_keep
        self.rasters_dir = rasters_dir
        self.viewshed_radius = viewshed_radius
        self.threads = threads
        self.viewshed_hw = viewshed_hw
        self.social_radius = social_radius
        self.num_ref_steps = num_ref_steps

        # Create mapping of target ID to reference indices
        self.id_to_ref_indices = self._create_id_to_ref_indices()

    def _create_id_to_ref_indices(self):
        id_to_ref_indices = {}
        for idx, row in self.reference_df.iterrows():
            target_id = row[self.target_id_col]
            if target_id not in id_to_ref_indices:
                id_to_ref_indices[target_id] = []
            id_to_ref_indices[target_id].append(idx)
        return id_to_ref_indices
            
    def __len__(self):
        return len(self.target_df)

    def __getitem__(self, idx):
        target_row = self.target_df.iloc[idx].copy()
        target_id = target_row[self.target_id_col]
        reference_indices = self.id_to_ref_indices.get(target_id, [])
        reference_rows = self.reference_df.iloc[reference_indices].copy()
        reference_rows = reference_rows.sample(self.num_ref_steps)
        observation_name = 'observation' + target_id.split('_')[0].split('b')[1]

        # Generate and downsample viewshed
        target_vis, target_vis_array = generate_downsample_viewshed(data_row = target_row,
                                                                    radius = self.viewshed_radius,
                                                                    threads = self.threads,
                                                                    metadata_df = self.metadata_df,
                                                                    observation_name = observation_name,
                                                                    rasters_dir = self.rasters_dir,
                                                                    viewshed_hw = self.viewshed_hw)
        target_row['visibility'] = target_vis
        target_row['vis_array'] = target_vis_array

        visibilities = []
        vis_arrays = []
        
        for r in np.arange(len(reference_rows)):
            row = reference_rows.iloc[r]
            ref_vis, ref_vis_array = generate_downsample_viewshed(data_row = row,
                                                                  radius = self.viewshed_radius,
                                                                  threads = self.threads,
                                                                  metadata_df = self.metadata_df,
                                                                  observation_name = observation_name,
                                                                  rasters_dir = self.rasters_dir,
                                                                  viewshed_hw = self.viewshed_hw)
            visibilities.append(ref_vis)
            vis_arrays.append(ref_vis_array)
        reference_rows['visibility'] = visibilities
        reference_rows['vis_array'] = vis_arrays

        # # Calculate social density
        # target_row['social_dens'] = sum(i < self.social_radius for i in target_row['neighbor_distances'])
        # reference_rows['social_dens'] = reference_rows['neighbor_distances'].apply(lambda x: sum(val < self.social_radius for val in x))

        # # Calculate proportion of group that is visible
        # target_row['social_vis'] = sum(i ==True for i in target_row['neighbor_visibility'])/sum(~np.isnan(i) for i in target_row['neighbor_visibility'])
        # reference_rows['social_vis'] = reference_rows['neighbor_visibility'].apply(lambda x: sum(val == True for val in x)/sum(~np.isnan(val) for val in x))
        
        # keep only specified columns and convert to dictionary
        if self.columns_to_keep:
            target_data = target_row[self.columns_to_keep].to_dict()
            reference_data = reference_rows[self.columns_to_keep].to_dict(orient='records')
        else:
            target_data = target_row.to_dict()
            reference_data = reference_rows.to_dict(orient = 'records')

        return target_data, reference_data

def custom_collate(batch):
    targets, references = zip(*batch)
    return targets, references

class ZebraBatchSampler(Sampler):
    def __init__(self, dataset, batch_size = 10):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_samples = len(dataset)

    def __iter__(self):
        for i in range(self.num_samples):
            yield [i]

    def __len__(self):
        return self.num_samples