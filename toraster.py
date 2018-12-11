import os
import numpy as np 
import arcpy
import sys

# set env variables
arcpy.env.overwriteOutput = True

# load the path in
path = sys.argv[1]
print "[*] path: " + path

# update filenames
filenames = list(map(lambda f: os.path.join(path, f), filter(lambda p: p.endswith("npz"), os.listdir(path))))

# convert all arrays to rasters
print "[*] converting arrays to rasters"
for f in filenames:
    try:
        array = np.load(f)['a']
        raster = arcpy.NumPyArrayToRaster(array, x_cell_size=30)
        raster.save(os.path.join(path, os.path.basename(f)+".tif"))
        print "[+] processed: " + f
    except:
        print "[!] failed to convert: " + f

