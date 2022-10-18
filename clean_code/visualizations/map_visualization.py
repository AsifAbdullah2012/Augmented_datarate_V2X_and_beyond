import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import matplotlib.pyplot as plt

from data import S3Measurements

bm = S3Measurements()
lis = bm.load_df()

print(' the size of the list is ', len(lis), ' round 1: ', len(lis[0]), 'round 2: ', len(lis[1]), 'round 3: ', len(lis[2]), 'round 4: ', len(lis[3]), 'round 5: ', len(lis[4]), 'round 6: ', len(lis[5]))
print('draw some thing ....')

