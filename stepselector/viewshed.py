from osgeo import gdal
import os
import numpy as np

# Define the function to generate the viewshed
def generate_viewshed(dsm_file, X, Y, height, targetRasterName, radius, threads):
    src_ds = gdal.Open(dsm_file)      
    srcBand = src_ds.GetRasterBand(1)
    c_options = ['NUM_THREADS=%i' %threads]
    
    gdal.ViewshedGenerate(
        srcBand=srcBand,
        driverName="GTIFF",
        targetRasterName=targetRasterName,
        creationOptions=c_options,
        observerX=X,
        observerY=Y,
        observerHeight=height,
        targetHeight=0,
        visibleVal=1,
        invisibleVal=0,
        outOfRangeVal=-10000,
        noDataVal=-10000,
        dfCurvCoeff=0.85714,
        mode=1,
        maxDistance=radius
    )
    src_ds = None


def generate_downsample_viewshed(data_row, radius, threads, metadata_df, observation_name, rasters_dir, viewshed_hw):
    X = data_row['lon']
    Y = data_row['lat']
    height = float(data_row['observer_height'])
    map_name = metadata_df[metadata_df['observation'] == observation_name]['big_map'].item() + '_dsm.tif'
    dsm = os.path.join(rasters_dir, 'DSMs', map_name)
    step_id = data_row['id']
    full_raster = '/vsimem/' + step_id + '_tempraster.tif'

    # generate full-resolution viewshed (stored temporarily in memory)
    generate_viewshed(dsm, X, Y, height, full_raster, radius, threads)

    # load the full-resolution viewshed and calculate the proportion of pixels that are visible
    vs = gdal.OpenEx(full_raster)
    vshed = vs.GetRasterBand(1)
    mean_full = vshed.GetStatistics(0,1)[2]

    # downsample the raster to the specified dimensions
    downsample_raster = '/vsimem/' + step_id + 'tempraster2.tif'
    kws = gdal.WarpOptions(
        format = 'GTiff',
        width = viewshed_hw,
        height = viewshed_hw,
        srcBands = [1],
        resampleAlg = 'average',
        outputType = gdal.GDT_Float64
    )
    gdal.Warp(
        destNameOrDestDS = downsample_raster,
        srcDSOrSrcDSTab = vs, options = kws)
    vs = None
    gdal.Unlink(full_raster)
    mvs = gdal.OpenEx(downsample_raster)

    # convert downsampled raster to numpy array
    viewshed_array = mvs.ReadAsArray(buf_type = gdal.GDT_Float64)
    mvs = None
    viewshed_array[viewshed_array == 5] = np.nan
    gdal.Unlink(downsample_raster)
    
    return mean_full, viewshed_array