import cfplot as cfp
import cf
import numpy as np
import pandas as pd
from datetime import timedelta
import matplotlib.pyplot as plt

# 1. Load the dataset
file_path = "~/Documents/Datasets/ERA5_monthly_averaged_SST.nc"
f = cf.read(file_path)
sst = f[0]  # Select the SST variable

# Collapse data by area mean (average over spatial dimensions)
am = sst.collapse("area: mean")  # equivalent to "X Y: mean"
am.squeeze(inplace=True)

# Check available coordinates (already found 'dimensioncoordinate0' as the time coordinate)
print("Available coordinates:", am.coordinates())

am_data_key = am.coordinate("dimensioncoordinate0", key=True)
am_copy = am.copy().subspace(**{am_data_key: cf.mam()})
print(am_copy)
exit()
# Retrieve the time coordinate using its internal identifier
am_data = am.coordinate("dimensioncoordinate0")
am_data_copy = am_copy.coordinate("dimensioncoordinate0")
# Convert the time data from 'hours since 1900-01-01' to datetime format
#time_units = time_coord.units
#am_time = time_coord.data
#am_time_datetime = am.subspace(T=cf.dt("2022-12-01 00:00:00"))

# Convert datetime to numeric values for plotting (e.g., using matplotlib date format)
#am_time_numeric = pd.to_numeric(pd.to_datetime(am_time_datetime))

# Extract data and replace missing values
#am_data = am.data
#am_data = np.where(am_data == -32767, np.nan, am_data)

# Create the line plot using the full dataset with numeric x-axis


print(am[:10])
print(am_data[:10].data.datetime_as_string)

cfp.gopen(file="Recipe3.png")  # Start the plot
cfp.lineplot(
    x=am_data,
    y=am,
    color="blue",
    title="Global Average Sea Surface Temperature",
    ylabel="Temperature (K)",
    xlabel="Time"
)

cfp.lineplot(
    x=am_data_copy,
    y=am_copy,
    color="red",
)
cfp.gclose()  # Finish the plot


# Save the plot to a file using matplotlib
plt.savefig("global_avg_sst_plot.png")

# Show the plot interactively
plt.show()

print("Plot created and saved successfully.")

