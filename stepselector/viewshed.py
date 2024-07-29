from osgeo import gdal

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