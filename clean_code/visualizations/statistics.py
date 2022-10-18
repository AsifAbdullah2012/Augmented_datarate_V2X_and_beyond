import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt

from data import S3Measurements

bm = S3Measurements()
lis = bm.load_df()


# draw some statistics firugres here ..........
print('statistics .... ')