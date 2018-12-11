# Read Me

Be mindful of RAM availability on the machine. You should have enough RAM such that this formula in bytes is at least 2GB below the machine limit:
grid = width * height * 16
bytes = (2 * years * grid) + (grid * num_eco_regions) + (num_cpus * 2 * grid)

# Inputs

Example inputs are provided in the example folder. All array data should have the same spatial extent, pixel size, height, and width.

## configuration format
The input configuration .toml file should have the following format.

input: the eclogical region input stored in npz format with signed 16bit integers.
distances: the csv table of ecological region codes that correspond to the pixel values of <input> and the pixels per year the region can move. It will have a header of "code,dist".
envelopes: the csv table of ecological region evelopes. It will have the header "code,temp_mx,temp_mn,precip_mx,precip_mn".
output_directory: the output directory for all of the output folders and files.
dispersion_directory: the directory to store the distance estimated dispersion results.
temperature_directory: the directory of temperature grids. Each grid should be named for the year it applies to in npz format (e.g. 0.npz = 0'th year). The data type of the pixels should be float32.
precipitation_directory: the directory of precipitation grids. Each grid should be named for the year it applies to in npz format (e.g. 0.npz = 0'th year). The data type of the pixels should be float32.
iters: the number of times to iterate the model.
years: number of years to run the model. It must correspond to the number of climate grids. There must be a single climate grid for each year.
seed: the seed value of the random number generator in the range [0,2^64).

This configuration information should be stored in a file named config.toml in the same path as the executable. It's context should follow the TOML format.
Here is an example.
'''
input="ecoregions.npz"
distances="distances.csv"
envelopes="envelopes.csv"
output_directory="./output"
dispersion_directory="./dispersion"
temperature_directory="./temp"
precipitation_directory="./precip"
iters=100
years=1
seed=10
'''

# Outputs
The model will produce an output grid for each iteration and year evaluated. It will contain the ecological region codes as pixel values in npz array format.
