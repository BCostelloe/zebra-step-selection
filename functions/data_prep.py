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