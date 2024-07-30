import math
import glob
import os
import numpy as np
import pandas as pd
from math import dist
import rasterio as rio
import pickle
from tqdm.notebook import tqdm
import utm
from osgeo import gdal

def extract_observed_steps(step_length, raw_tracks_directory, save_directory, obs_to_process = None):
    """
    Spatially discretizes observed trajectories

    Parameters:
        - step_length: the minimum step length (in meters)
        - raw_tracks_directory: folder where trajectory data for each observation is stored. Should be one .npy per observation.
        - save_directory: where to save the resulting dataframes.
        - obs_to_process (OPTIONAL): if you don't want to process all trajectory data, give a list of observations, e.g. ['ob015', 'ob074']
    """
    # Get files to process
    if obs_to_process is None:
        raw_tracks_files = sorted(glob.glob(os.path.join(raw_tracks_directory, '*.npy')))
    else:
        raw_tracks_files = []
        for o in obs_to_process:
            files = sorted(glob.glob(os.path.join(raw_tracks_directory, '%s*.npy' %o)))
            raw_tracks_files.extend(files)
    ## good up to here, need to edit rest of function
    for f in tqdm(raw_tracks_files):
        file = np.load(f, allow_pickle = True)
        for t, track in enumerate(file):
            new_filename = str(f.split('/')[-1].split('_')[0] + '_track' + '{:02}'.format(t) + '_%imsteps.pkl' %step_length)
            obs = f.split('/')[-1].split('_')[0]
            new_file = os.path.join(save_directory, new_filename)
            steps = []
            frames = []
            first_step = np.min(np.where(np.isfinite(track))[0])
            ref_point = track[first_step]
            steps.append(ref_point)
            frames.append(first_step)
            for n, i in enumerate(track):
                if dist(i, ref_point) > step_length:
                    ref_point = i
                    steps.append(ref_point)
                    frames.append(n)
            lons = [i[0] for i in steps]
            lats = [i[1] for i in steps]
            ids = [str(obs + '_' + str(t) + '_f' + str(p) + '_ob') for p in frames]
            new_df = pd.DataFrame(zip(frames, lats, lons, ids, ids), columns = ['frame', 'lat', 'lon', 'target_id', 'id'])
            new_df.to_pickle(new_file)

def calculate_full_angles(points): # from ChatGPT
    """
    Calculates the turning angles between vectors defined by a series of location coordinates.
    """
    angles = []
    vectors = []

    # Compute vectors between consecutive points
    for i in range(len(points) - 1):
        vector = np.array([points[i+1][0] - points[i][0], points[i+1][1] - points[i][1]])
        vectors.append(vector)

    # Compute angles between consecutive vectors
    for i in range(len(vectors) - 1):
        vector1 = vectors[i]
        vector2 = vectors[i+1]

        # Calculate dot product and magnitudes
        dot_product = np.dot(vector1, vector2)
        mag1 = np.linalg.norm(vector1)
        mag2 = np.linalg.norm(vector2)

        # Calculate the angle in radians
        if mag1 * mag2 == 0:  # prevent division by zero if any vector is a zero vector
            angle = 0
        else:
            cos_angle = dot_product / (mag1 * mag2)
            # Ensure the cosine value is within the valid range to avoid numerical issues
            cos_angle = max(min(cos_angle, 1), -1)
            angle = np.arccos(cos_angle)

        # Determine the direction of the angle (clockwise or counterclockwise)
        cross_product = np.cross(vector1, vector2)
        if cross_product > 0:
            # Counterclockwise rotation
            angle = angle
        else:
            # Clockwise rotation
            #angle = 2 * np.pi - angle
            angle = -angle

        angle = np.degrees(angle)  # Convert to degrees
        angles.append(angle)

    return angles

# This is adapted from ChatGPT.

def calculate_multiple_simulated_points(points, data_frame, raster_band, num_simulated_points, originX, originY, cellSizeX, cellSizeY):
    """
    Calculate multiple new points from given points, with specified numbers of simulated points.
    Random distances and angles are drawn from the provided DataFrame.

    To implement: Points are checked to ensure they are within the bounds of a DSM raster. 
    Points outside the bounds are discarded and new ones drawn.
    
    :param points: List of tuples (x, y) representing coordinates.
    :param data_frame: DataFrame with columns 'angle' and 'distance' to draw random samples.
    :param num_simulated_points: Number of simulated points to generate per real point.
    :param raster_band: band from DSM raster
    
    :return: Dictionary with each real point as a key, and list of tuples (x, y) of simulated points.
    """
    simulated_points = {}

    for i in range(1, len(points)):
        simulated_points_for_point = []
        # Calculate the vector from the previous point to the current point
        dx = points[i][0] - points[i-1][0]
        dy = points[i][1] - points[i-1][1]
        
        # Calculate the angle of this vector from the horizontal
        vector_angle = np.degrees(np.arctan2(dx, dy))
        
        counter = num_simulated_points
        
        while counter > 0:
            # Draw random distance and angle
            sample = data_frame.sample(1).iloc[0]
            distance = sample['distance']
            angle = sample['angle']
            
            # Calculate the absolute angle for the new simulated point
            current_angle = vector_angle + angle
            radian_angle = np.radians(current_angle)
            
            # Calculate the new simulated point's coordinates
            new_y = points[i-1][1] + distance * np.cos(radian_angle)
            new_x = points[i-1][0] + distance * np.sin(radian_angle)

            ## Check that the simulated point has a valid value in the DSM
            col = int((new_x - originX)/cellSizeX)
            row = int((new_y - originY)/cellSizeY)
            DSM_val = raster_band[row,col]
            if DSM_val == -10000:
                continue
            else:
                simulated_points_for_point.append((new_x, new_y))
                counter = counter - 1
        
        simulated_points[points[i]] = simulated_points_for_point
    
    return simulated_points

def simulate_fake_steps(n_steps, observed_steps_directory, save_directory, rasters_directory, ob_metadata_file, obs_to_process = None):
    """
    Generate some number of simulated alternative steps for each observed step.
    Simulated steps are defined by drawing randomly from the observed distributions of step lengths and turning angles.
    Each step is checked to ensure that it falls within the bounds of the associated DSM raster.

    Parameters:
        - n_steps: the number of simulated steps that should be generated for each observed step
        - observed_steps_directory: folder where the observed steps .pkl files are stored. Should be one .pkl per track.
        - save_directory: where to save the resulting dataframes.
        - rasters_directory: folder where raster files are stored
        - ob_metadata_file: file giving the map area names that correspond to each observation
        - obs_to_process (OPTIONAL): if you don't want to process all trajectory data, give a list of observations, e.g. ['ob015', 'ob074']
    """
    # Load metadata file
    metadata = pd.read_csv(ob_metadata_file)
    
    # Define observations to be processed
    if obs_to_process is None:
        observed_step_files = sorted(glob.glob(os.path.join(observed_steps_directory, '*.pkl')))
        observations = []
        for f in observed_step_files:
            obs = f.split('/')[-1].split('_')[0]
            observations = np.append(observations, obs)
            #observations.append(obs)
            observations = np.unique(observations)
    else:
        observations = obs_to_process


    for o in tqdm(observations):
        # get map name and info
        full_ob_name = 'observation' + o.split('b')[-1]
        map_name = metadata[metadata['observation'] == full_ob_name]['big_map'].item()
        dsm_file = os.path.join(rasters_directory, 'DSMs', map_name + '_dsm.tif')
        DSM = rio.open(dsm_file)
        dsm = DSM.read(1)
        originX = DSM.bounds[0]
        originY = DSM.bounds[3]
        cellSizeX = DSM.transform[0]
        cellSizeY = DSM.transform[4]

        # get observed step files to process
        observed_step_files = sorted(glob.glob(os.path.join(observed_steps_directory, '%s*.pkl' %o)))

        # process each observed step file
        for f in observed_step_files:
            step_file = pd.read_pickle(f)
            #obs = f.split('/')[-1].split('_')[0]
            track = f.split('/')[-1].split('_')[1]
    
            # get observed step locations
            points = step_file[['lon', 'lat']].to_numpy()
            if len(points) > 2:
            
                # calculate step lengths and angles
                dists = []
                for n in range(1, len(points)-1):
                    new_dist = dist(points[n], points[n+1])
                    dists.append(new_dist)
                angles = calculate_full_angles(points)
                step_lengths_angles = pd.DataFrame(zip(angles, dists), columns = ['angle', 'distance'])
                
                # generate n_steps simulated steps for each observed step
                list_points = list(map(tuple,points))
                simulated_points = calculate_multiple_simulated_points(list_points, step_lengths_angles, dsm, n_steps, 
                                                                       originX = originX,
                                                                       originY = originY,
                                                                       cellSizeX = cellSizeX,
                                                                       cellSizeY = cellSizeY)
                
                # combine everything into the same dataframe and save
                times = []
                lats = []
                lons = []
        
                for ob in simulated_points.keys():
                    for i in simulated_points[ob]:
                        lats.append(i[1])
                        lons.append(i[0])
                        time = step_file.loc[((step_file['lat'] == ob[1]) & (step_file['lon'] == ob[0])), 'frame'].item()
                        times.append(time)
            
                new_df = pd.DataFrame(zip(times, lats, lons), columns = ['frame', 'lat', 'lon'])
                new_df['step_type'] = 'simulated'
                new_df['count'] = new_df.groupby('frame').cumcount()
                new_df['id'] = [str(o + '_' + track + '_f' + str(frame) + '_sim-' + str(count)) for frame, count in zip(new_df['frame'],new_df['count'])]
                new_df['target_id'] = [str(o + '_' + track + '_f' + str(frame) + '_ob') for frame in new_df['frame']]
                new_df.drop(['count'], inplace=True, axis=1)
                #full_df = pd.concat([step_file, new_df], ignore_index = True)
                
                # create new filename
                new_filename = f.split('/')[-1].split('.')[0] + '_sim.pkl'
                new_file = os.path.join(save_directory, new_filename)
                new_df.to_pickle(new_file)
        DSM.close()

# source: https://stackoverflow.com/questions/28260962/calculating-angles-between-line-segments-python-with-math-atan2
def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]

# source: https://stackoverflow.com/questions/28260962/calculating-angles-between-line-segments-python-with-math-atan2
def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360
    
    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 
        
        return ang_deg

def get_observer_and_step_info(observed_steps_directory, simulated_steps_directory, ob_metadata_file, obs_to_process = None):
    """
    Generate some number of simulated alternative steps for each observed step.
    Simulated steps are defined by drawing randomly from the observed distributions of step lengths and turning angles.
    Each step is checked to ensure that it falls within the bounds of the associated DSM raster.

    Parameters:
        - observed_steps_directory: folder where the observed steps .pkl files are stored. Should be one .pkl per track.
        - simulated_steps_directory: folder where the simulated steps .pkl files are stored. Should be one .pkl per track.
        - ob_metadata_file: file giving the map area names that correspond to each observation
        - obs_to_process (OPTIONAL): if you don't want to process all trajectory data, give a list of observations, e.g. ['ob015', 'ob074']
    """
    # Load metadata file
    meta = pd.read_csv(ob_metadata_file)
    
    # Define observations to be processed
    if obs_to_process is None:
        observed_step_files = sorted(glob.glob(os.path.join(observed_steps_directory, '*.pkl')))
        observations = []
        for f in observed_step_files:
            obs = f.split('/')[-1].split('_')[0]
            observations = np.append(observations, obs)
            #observations.append(obs)
            observations = np.unique(observations)
    else:
        observations = obs_to_process

    for o in tqdm(observations):
        # get observed step files to process
        observed_step_files = sorted(glob.glob(os.path.join(observed_steps_directory, '%s*.pkl' %o)))
        simulated_step_files = sorted(glob.glob(os.path.join(simulated_steps_directory, '%s*.pkl' %o)))
        step_files = observed_step_files + simulated_step_files

        # process each observed step file
        for f in step_files:
            data = pd.read_pickle(f)
            data.reset_index(drop = True, inplace = True)
            obnum = o.split('b')[-1]
            obname = str('observation' + obnum)
            tracknum = int(f.split('/')[-1].split('_')[1].split('k')[-1])
            step_type = f.split('/')[-2]

            # Get lat and lon of observer and convert to UTMs
            ob_lat = meta[meta['observation'] == obname]['observer_lat'].iloc[0]
            ob_lon = meta[meta['observation'] == obname]['observer_lon'].iloc[0]
            oblon, oblat, N, L = utm.from_latlon(ob_lat, ob_lon)
            data['observers_lat'] = oblat
            data['observers_lon'] = oblon
            
            # Calculate angle between step vector and vector to observation team
            ## First, get a list of all unique time values in order
            times = np.sort(data.frame.unique())
            
            ## create column to store result
            data['angle_to_observers'] = np.nan
            data['delta_observer_dist'] = np.nan
            data['dist_to_observer'] = np.nan
            data['prev_step'] = None
            data['step_length_m'] = np.nan
            data['step_duration_s'] = np.nan
            data['step_speed_mps'] = np.nan
            
            for t in np.arange(len(times)):
                for index, row in data.iterrows():
                    if row['frame'] == times[t]:
        
                        # calculate distance to observer team
                        step_end_lat = row['lat']
                        step_end_lon = row['lon']
                        dist_end = dist((step_end_lon, step_end_lat), (oblon, oblat))
                        data.loc[index, 'dist_to_observer'] = dist_end
        
                        # the following operations can't be done for the first location since there was no previous location
                        if t > 0:

                            if step_type == 'observed':
                        
                                # define vectors representing current step and vector to observers
                                start_lat = data[data['frame'] == times[t-1]]['lat'].iloc[0]
                                start_lon = data[data['frame'] == times[t-1]]['lon'].iloc[0]
                                start_step = data[data['frame'] == times[t-1]]['id'].iloc[0]

                            if step_type == 'simulated':
                                ref_steps_file = '%s_track%s*.pkl' %(o, "{:02d}".format(tracknum))
                                ref_steps_path = glob.glob(os.path.join(observed_steps_directory, ref_steps_file))[0]
                                ref_steps = pd.read_pickle(ref_steps_path)
                                start_lat = ref_steps[ref_steps['frame'] == times[t-1]]['lat'].iloc[0]
                                start_lon = ref_steps[ref_steps['frame'] == times[t-1]]['lon'].iloc[0]
                                start_step = ref_steps[ref_steps['frame'] == times[t-1]]['id'].iloc[0]

                            step_vect = ((start_lon,start_lat),(step_end_lon,step_end_lat))
                            ob_vect = ((start_lon,start_lat),(oblon,oblat))
                            
                            angle = ang(step_vect, ob_vect)
                            
                            data.loc[index, 'angle_to_observers'] = angle
            
                            # Calculate change in distance to the observer team
                            dist_start = dist((start_lon, start_lat), (oblon, oblat))
                            
                            delta_observer_distance = dist_end - dist_start
                            data.loc[index, 'delta_observer_dist'] = delta_observer_distance
                            
            
                            # While we're here, let's calculate the step length, time duration and speed
                            step_length_m = dist((start_lon, start_lat), (step_end_lon, step_end_lat))
                            start_frame = times[t-1]
                            step_duration_s = (times[t] - times[t-1])/30
                            step_speed_mps = step_length_m/step_duration_s
            
                            data.loc[index, 'prev_step'] = start_step
                            data.loc[index, 'step_length_m'] = step_length_m
                            data.loc[index, 'step_duration_s'] = step_duration_s
                            data.loc[index, 'step_speed_mps'] = step_speed_mps

            data = data[['frame', 'lat', 'lon', 'id', 'target_id', 'angle_to_observers', 'dist_to_observer', 'delta_observer_dist', 'prev_step', 
                        'step_length_m', 'step_duration_s', 'step_speed_mps']]
            data.to_pickle(f)


def calculate_zebra_heights(observed_steps_directory, simulated_steps_directory, rasters_directory, ob_metadata_file, obs_to_process = None):
    """
    Calculate the elevation of a zebra's head (assumed to be 1.5 meters tall) relative to the surface of the DSM by sampling a pre-generated
    zebra height raster.

    Parameters:
        - observed_steps_directory: folder where the observed steps .pkl files are stored. Should be one .pkl per track.
        - simulated_steps_directory: folder where the simulated steps .pkl files are stored. Should be one .pkl per track.
        - rasters_directory: folder where raster files are stored
        - ob_metadata_file: file giving the map area names that correspond to each observation
        - obs_to_process (OPTIONAL): if you don't want to process all trajectory data, give a list of observations, e.g. ['ob015', 'ob074']
    """
    # Load metadata file
    metadata = pd.read_csv(ob_metadata_file)
    
    # Define observations to be processed
    if obs_to_process is None:
        observed_step_files = sorted(glob.glob(os.path.join(observed_steps_directory, '*.pkl')))
        observations = []
        for f in observed_step_files:
            obs = f.split('/')[-1].split('_')[0]
            observations = np.append(observations, obs)
            observations = np.unique(observations)
    else:
        observations = obs_to_process


    for o in tqdm(observations):
        # get map name and info
        full_ob_name = 'observation' + o.split('b')[-1]
        map_name = metadata[metadata['observation'] == full_ob_name]['big_map'].item()
        obheights_raster = os.path.join(rasters_directory, 'zebra_heights', '%s_ZebraHeights_1-5m.tif' %map_name)
        obheights_raster = rio.open(obheights_raster)
        obheights = obheights_raster.read(1)
        originX = obheights_raster.bounds[0]
        originY = obheights_raster.bounds[3]
        cellSizeX = obheights_raster.transform[0]
        cellSizeY = obheights_raster.transform[4]

        # get step files
        observed_step_files = sorted(glob.glob(os.path.join(observed_steps_directory, '%s*.pkl' %o)))
        simulated_step_files = sorted(glob.glob(os.path.join(simulated_steps_directory, '%s*.pkl' %o)))
        step_files = observed_step_files + simulated_step_files

        for f in step_files:
            steps = pd.read_pickle(f)
            cols = [int((x - originX)/cellSizeX) for x in steps['lon']]
            rows = [int((y - originY)/cellSizeY) for y in steps['lat']]
            steps['observer_height'] = [obheights[row,col] for row,col in zip(rows, cols)]

            steps.to_pickle(f)
        obheights_raster.close()

def road_or_no(observed_steps_directory, simulated_steps_directory, rasters_directory, ob_metadata_file, obs_to_process = None):
    """
    Calculate sample a pre-generated roads raster to determine if a given point is on a rod

    Parameters:
        - observed_steps_directory: folder where the observed steps .pkl files are stored. Should be one .pkl per track.
        - simulated_steps_directory: folder where the simulated steps .pkl files are stored. Should be one .pkl per track.
        - rasters_directory: folder where raster files are stored
        - ob_metadata_file: file giving the map area names that correspond to each observation
        - obs_to_process (OPTIONAL): if you don't want to process all trajectory data, give a list of observations, e.g. ['ob015', 'ob074']
    """
    # Load metadata file
    metadata = pd.read_csv(ob_metadata_file)
    
    # Define observations to be processed
    if obs_to_process is None:
        observed_step_files = sorted(glob.glob(os.path.join(observed_steps_directory, '*.pkl')))
        observations = []
        for f in observed_step_files:
            obs = f.split('/')[-1].split('_')[0]
            observations = np.append(observations, obs)
            observations = np.unique(observations)
    else:
        observations = obs_to_process


    for o in tqdm(observations):
        # get map name and info
        full_ob_name = 'observation' + o.split('b')[-1]
        map_name = metadata[metadata['observation'] == full_ob_name]['big_map'].item()
        roads_raster_file = os.path.join(rasters_directory, 'roads', '%s_roadsraster.tif' %map_name)
        roads_raster = rio.open(roads_raster_file)
        roads = roads_raster.read(1)
        originX = roads_raster.bounds[0]
        originY = roads_raster.bounds[3]
        cellSizeX = roads_raster.transform[0]
        cellSizeY = roads_raster.transform[4]

        # get step files
        observed_step_files = sorted(glob.glob(os.path.join(observed_steps_directory, '%s*.pkl' %o)))
        simulated_step_files = sorted(glob.glob(os.path.join(simulated_steps_directory, '%s*.pkl' %o)))
        step_files = observed_step_files + simulated_step_files

        for f in step_files:
            steps = pd.read_pickle(f)
            cols = [int((x - originX)/cellSizeX) for x in steps['lon']]
            rows = [int((y - originY)/cellSizeY) for y in steps['lat']]
            steps['road'] = [roads[row,col] for row,col in zip(rows, cols)]
            steps['road'] = steps['road'].replace(255, 1)
            steps['road'] = steps['road'].astype('int32')

            steps.to_pickle(f)
        roads_raster.close()

def step_slope(observed_steps_directory, simulated_steps_directory, rasters_directory, ob_metadata_file, obs_to_process = None):
    """
    Calculate the slope angle of each step

    Parameters:
        - observed_steps_directory: folder where the observed steps .pkl files are stored. Should be one .pkl per track.
        - simulated_steps_directory: folder where the simulated steps .pkl files are stored. Should be one .pkl per track.
        - rasters_directory: folder where the raster files are stored
        - ob_metadata_file: file giving the map area names that correspond to each observation
        - obs_to_process (OPTIONAL): if you don't want to process all trajectory data, give a list of observations, e.g. ['ob015', 'ob074']
    """
    # Load metadata file
    metadata = pd.read_csv(ob_metadata_file)
    
    # Define observations to be processed
    if obs_to_process is None:
        observed_step_files = sorted(glob.glob(os.path.join(observed_steps_directory, '*.pkl')))
        observations = []
        for f in observed_step_files:
            obs = f.split('/')[-1].split('_')[0]
            observations = np.append(observations, obs)
            observations = np.unique(observations)
    else:
        observations = obs_to_process

        
    for o in tqdm(observations):
        # get map name and info
        full_ob_name = 'observation' + o.split('b')[-1]
        map_name = metadata[metadata['observation'] == full_ob_name]['big_map'].item()
        DTM = os.path.join(rasters_directory, 'DTMS', '%s_dtm.tif' %map_name)
        dtm = rio.open(DTM)
        alts = dtm.read(1)
        min_x, min_y, max_x, max_y = dtm.bounds

        # get step files
        observed_step_files = sorted(glob.glob(os.path.join(observed_steps_directory, '%s*.pkl' %o)))
        simulated_step_files = sorted(glob.glob(os.path.join(simulated_steps_directory, '%s*.pkl' %o)))
        step_files = observed_step_files + simulated_step_files

        for f in step_files:
            steps = pd.read_pickle(f)
            steps.reset_index(drop = True, inplace = True)
            step_type = f.split('/')[-2]
            tracknum = int(f.split('/')[-1].split('_')[1].split('k')[-1])
            if step_type == 'simulated':    
                ref_steps_file = '%s_track%s*.pkl' %(o, "{:02d}".format(tracknum))
                ref_steps_path = glob.glob(os.path.join(observed_steps_directory, ref_steps_file))[0]
                ref_steps = pd.read_pickle(ref_steps_path)
            times = np.sort(steps.frame.unique())
            steps['ground_slope'] = np.nan

            for t in np.arange(1,len(times)):
                for index, row in steps.iterrows():
                    # get coordinates for the start and end point of the step
                    if row['frame'] == times[t]:
                        end_y = row['lat']
                        end_x = row['lon']
                        end_point = (end_x, end_y)

                        if step_type == 'observed':
                            start_y = steps[steps['frame'] == times[t-1]]['lat'].iloc[0]
                            start_x = steps[steps['frame'] == times[t-1]]['lon'].iloc[0]
                            start_point = (start_x, start_y)

                        if step_type == 'simulated':
                            start_y = ref_steps[ref_steps['frame'] == times[t-1]]['lat'].iloc[0]
                            start_x = ref_steps[ref_steps['frame'] == times[t-1]]['lon'].iloc[0]
                            start_point = (start_x, start_y)

                    
                        if ((start_x >= min_x) & (start_x <= max_x) &
                            (end_x >= min_x) & (end_x <= max_x) &
                            (start_y >= min_y) & (start_y <= max_y) &
                            (end_y >= min_y) & (end_y <= max_y)):
                        
                            # get altitudes at each point and calculate the difference between them. 
                            start_row, start_col = rio.transform.rowcol(dtm.transform, start_x, start_y)
                            start_alt = alts[start_row, start_col]
                            #print('Altitude at start point is ', start_alt)
        
                            end_row, end_col = rio.transform.rowcol(dtm.transform, end_x, end_y)
                            end_alt = alts[end_row, end_col]
                            #print('Altitude at end point is ', end_alt)
        
                            alt_change = end_alt - start_alt
                            # if end_alt > start_alt:
                            #     print('end point is higher; slope should be positive')
                            # else:
                            #     print('end point is lower; slope should be negative')
        
                            # calculate distance between start and end point
                            distance = math.dist(start_point, end_point)
                            #print('Distance is ', distance)
        
                            # Use altitude difference and step length to calculate the terrain slope. If the start point is lower
                            # than the end point, the slope should be positive; if the start point is higher than the end point, the
                            # slope should be negative
                            slope = math.degrees(math.atan(alt_change/distance))
                            # print('slope is ', slope)
        
                            steps.loc[index, 'ground_slope'] = slope
                        else:
                            continue
            steps.to_pickle(f)
        dtm.close()

def get_social_info(observed_steps_directory, simulated_steps_directory, raw_tracks_directory, rasters_directory, ob_metadata_file, track_metadata_file, obs_to_process = None):
    """
    Get information on the focal animal's relationship (distance, visibility) to other group mates at each step location

    Parameters:
        - observed_steps_directory: folder where the observed steps .pkl files are stored. Should be one .pkl per track.
        - simulated_steps_directory: folder where the simulated steps .pkl files are stored. Should be one .pkl per track.
        - raw_tracks_directory: folder where the raw trajectories are stored. Should be one .npy per observation.
        - rasters_directory: folder where raster files are stored
        - ob_metadata_file: file giving the map area names that correspond to each observation
        - obs_to_process (OPTIONAL): if you don't want to process all trajectory data, give a list of observations, e.g. ['ob015', 'ob074']
    """
    # Load metadata file
    metadata = pd.read_csv(ob_metadata_file)
    track_metadata = pd.read_csv(track_metadata_file)
    
    # Define observations to be processed
    if obs_to_process is None:
        observed_step_files = sorted(glob.glob(os.path.join(observed_steps_directory, '*.pkl')))
        observations = []
        for f in observed_step_files:
            obs = f.split('/')[-1].split('_')[0]
            observations = np.append(observations, obs)
            observations = np.unique(observations)
    else:
        observations = obs_to_process

    for o in tqdm(observations):
        # get raw tracks file
        raw_tracks_file = os.path.join(raw_tracks_directory, str(o + '_utm_tracks.npy'))
        raw_tracks = np.load(raw_tracks_file, allow_pickle = True)
        
        # get map name
        full_ob_name = 'observation' + o.split('b')[-1]
        map_name = metadata[metadata['observation'] == full_ob_name]['big_map'].item()

        # get zebra heights raster & info
        obheights_raster = os.path.join(rasters_directory, 'zebra_heights', '%s_ZebraHeights_1-5m.tif' %map_name)
        obheights_raster = rio.open(obheights_raster)
        obheights = obheights_raster.read(1)
        originX = obheights_raster.bounds[0]
        originY = obheights_raster.bounds[3]
        cellSizeX = obheights_raster.transform[0]
        cellSizeY = obheights_raster.transform[4]

        # get DSM raster and info
        dsm_raster = os.path.join(rasters_directory, 'DSMs', '%s_dsm.tif' %map_name)
        dsm = rio.open(dsm_raster)
        dsm_rio = dsm.read(1)
        dsmX = dsm.bounds[0]
        dsmY = dsm.bounds[3]
        dsmcellSizeX = dsm.transform[0]
        dsmcellSizeY = dsm.transform[4]
        dsm.close()

        dataset = gdal.Open(dsm_raster)
        dsm = dataset.GetRasterBand(1)

        # get step files
        observed_step_files = sorted(glob.glob(os.path.join(observed_steps_directory, '%s*.pkl' %o)))
        simulated_step_files = sorted(glob.glob(os.path.join(simulated_steps_directory, '%s*.pkl' %o)))
        step_files = observed_step_files + simulated_step_files

        for f in step_files:
            steps = pd.read_pickle(f)
            track = int(f.split('/')[-1].split('_')[1].split('k')[1])
            frames = list(steps.frame)
            points = list(zip(steps.lon, steps.lat))
            social_dat = []
            for n, p in enumerate(frames):
                focal_point = points[n]
                step_id = steps.loc[n,'id']
                neighbor_ids = []
                neighbor_spps = []
                neighbor_points = []
                neighbor_distances = []
                neighbor_visibilities = []
                focal_height = steps.loc[n, 'observer_height']

                # calculate focal animal's altitude in DSM units
                dsm_col_focal = int((focal_point[0] - dsmX)/dsmcellSizeX)
                dsm_row_focal = int((focal_point[1] - dsmY)/dsmcellSizeY)
                focal_altitude = focal_height + float(dsm_rio[dsm_row_focal, dsm_col_focal])

                # Get info on neighbors from raw tracks
                for t in np.arange(len(raw_tracks)):
                    if t == track:
                        continue
                    else:
                        neighbor_id = track_metadata[(track_metadata['observation']==full_ob_name) & (track_metadata['track'] == t)]['individual_ID'].item() #look this up from track metadata based on track number
                        neighbor_spp = track_metadata[(track_metadata['observation']==full_ob_name) & (track_metadata['track'] == t)]['species'].item()
                        neighbor_point = raw_tracks[t][p]
        
                        if not np.isnan(neighbor_point).any():  
                            neighbor_dist = math.dist([neighbor_point[0], neighbor_point[1]], [focal_point[0], focal_point[1]])
                            # get neighbor height by sampling zebra heights raster
                            col = int((neighbor_point[0] - originX)/cellSizeX)
                            row = int((neighbor_point[1] - originY)/cellSizeY)
                            neighbor_height = float(obheights[row,col])
        
                            dsm_col_neighbor = int((neighbor_point[0] - dsmX)/dsmcellSizeX)
                            dsm_row_neighbor = int((neighbor_point[1] - dsmY)/dsmcellSizeY)
                            neighbor_altitude = neighbor_height + float(dsm_rio[dsm_row_neighbor, dsm_col_neighbor])
                            neighbor_vis = gdal.IsLineOfSightVisible(band = dsm,
                                                                    xA = dsm_col_focal,
                                                                    yA = dsm_row_focal,
                                                                    zA = focal_altitude,
                                                                    xB = dsm_col_neighbor,
                                                                    yB = dsm_row_neighbor,
                                                                    zB = neighbor_altitude)
                            is_visible = neighbor_vis.is_visible
                            neighbor_ids.append(neighbor_id)
                            neighbor_spps.append(neighbor_spp)
                            neighbor_points.append(neighbor_point)
                            neighbor_distances.append(neighbor_dist)
                            neighbor_visibilities.append(is_visible)
                        else:
                            neighbor_ids.append(neighbor_id)
                            neighbor_spps.append(neighbor_spp)
                            neighbor_points.append(neighbor_point)
                            neighbor_distances.append(np.nan)
                            neighbor_visibilities.append(np.nan)
            
                # Create dictionary with keys for step ID, neighbor IDs, neighbor distances, neighbor visibilities 
                social_dict = {'step_id': step_id, 
                               'neighbor_ids': neighbor_ids, 
                               'neighbor_spps': neighbor_spps,
                               'neighbor_points': neighbor_points, 
                               'neighbor_distances': neighbor_distances,
                               'neighbor_visibility': neighbor_visibilities
                              }
                    # Store dictionary in social_dat list
                social_dat.append(social_dict)
                # Define filename for list of dictionaries - one list per track, one dictionary per step location
            steps['neighbor_ids'] = [x['neighbor_ids'] if x['step_id'] == y else np.nan for x, y in zip(social_dat, steps['id'])]
            steps['neighbor_spps'] = [x['neighbor_spps'] if x['step_id'] == y else np.nan for x, y in zip(social_dat, steps['id'])]
            steps['neighbor_points'] = [x['neighbor_points'] if x['step_id'] == y else np.nan for x, y in zip(social_dat, steps['id'])]
            steps['neighbor_distances'] = [x['neighbor_distances'] if x['step_id'] == y else np.nan for x, y in zip(social_dat, steps['id'])]
            steps['neighbor_visibility'] = [x['neighbor_visibility'] if x['step_id'] == y else np.nan for x, y in zip(social_dat, steps['id'])]
            
            steps.to_pickle(f)
        obheights_raster.close()
        dataset = None

### I don't think the things below this line are used in the pipeline:
def calculate_initial_compass_bearing(pointA, pointB): # adjusted from source: https://gist.github.com/jeromer/2005586
    """
    Calculates the bearing between two points.
    The formulae used is the following:
        θ = atan2(sin(Δlong).cos(lat2),
                  cos(lat1).sin(lat2) − sin(lat1).cos(lat2).cos(Δlong))
    :Parameters:
      - `pointA: The tuple representing the latitude/longitude for the
        first point. Latitude and longitude must be in decimal degrees
      - `pointB: The tuple representing the latitude/longitude for the
        second point. Latitude and longitude must be in decimal degrees
    :Returns:
      The bearing in degrees
    :Returns Type:
      float
    """
    if (type(pointA) != tuple) or (type(pointB) != tuple):
        raise TypeError("Only tuples are supported as arguments")

    lat1 = math.radians(pointA[0])
    lat2 = math.radians(pointB[0])

    diffLong = math.radians(pointB[1] - pointA[1])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
            * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)

    # # Now we have the initial bearing but math.atan2 return values
    # # from -180° to + 180° which is not what we want for a compass bearing
    # # The solution is to normalize the initial bearing as shown below
    initial_bearing = math.degrees(initial_bearing)
    # compass_bearing = (initial_bearing + 360) % 360

    return initial_bearing
    # return compass_bearing