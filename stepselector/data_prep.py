import math
import glob
import os
import numpy as np
import pandas as pd
from math import dist

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
    for f in raw_tracks_files:
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
            new_df = pd.DataFrame(zip(frames, lats, lons, ids), columns = ['frame', 'lat', 'lon', 'id'])
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
        vector_angle = np.degrees(np.arctan2(dy, dx))
        
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
            new_y = points[i-1][0] + distance * np.cos(radian_angle)
            new_x = points[i-1][1] + distance * np.sin(radian_angle)

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