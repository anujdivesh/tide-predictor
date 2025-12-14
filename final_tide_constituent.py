import json
import numpy as np
import os
import sys
from tideAnalysis.utils.pytmd_tide_model import TideModel

# Function to get constituents for any lat/lon
def get_constituents_by_latlon(lat, lon):
    TideSolver = TideModel()
    lonGrid, latGrid = TideSolver.get_model_nearest_point(lon, lat)
    amp, ph, c = TideSolver.Model.extract_constants(lonGrid, latGrid, **{'extrapolate': True})
    amp = amp.data.squeeze()
    ph = ph.data.squeeze()
    vals = { name: {'amplitude': float(amp[ii]), 'phase': float(ph[ii])} for ii, name in enumerate(c) }
    return vals


# Example usage
if __name__ == "__main__":
    # Example: replace with your desired lat/lon
    lat = -21.20475
    lon = -159.784777777778
    result = get_constituents_by_latlon(lat, lon)
    print(json.dumps(result, indent=4))

# # Print them for manual check
# txt = '{c}: {a:.5f} - {p:.5f}'

# for ii, ci in enumerate(c):
#     print('-'*20)
#     print(txt.format(c=ci, a=amp[ii], p=ph[ii]))

