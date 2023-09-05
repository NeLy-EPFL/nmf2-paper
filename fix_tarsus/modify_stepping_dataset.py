import pickle
from pathlib import Path
import pkg_resources
import numpy as np
import copy

# Load the data
# Load recorded data
data_path = Path(pkg_resources.resource_filename('flygym', 'data'))
with open(data_path / 'behavior' / 'single_steps.pkl', 'rb') as f:
    data = pickle.load(f)
print(data["meta"])

new_data = {}
old_data = {}
for key, item in data.items():
    new_data[key] = item.copy()
    old_data[key] = item.copy()

assert np.abs(np.mean(data[f"joint_RFTarsus1"]) - -1*0.727255077713601) < 1e-4, "The data was already modified with respect" \
                                                          " to the orignal stepping data" \
                                                          "There is no guarantee thos changes will make sense"

#shift RF and LF Tarsus angles by 30degrees
for leg in ["RF", "LF"]:
    new_data[f"joint_{leg}Tarsus1"] += np.deg2rad(30)

old_path = data_path / 'behavior' / 'single_steps_old.pkl'
# check old stepping data does not exist
"""assert old_path.exists() == False, "The data was already modified by modifying it again you would change the data twice " \
                                   "(60 degres shift of the forn leg tarsusangles)"
                                   """

# save the old data
with open(old_path, 'wb') as f:
    pickle.dump(old_data, f)
# save the new data
with open(data_path / 'behavior' / 'single_steps.pkl', 'wb') as f:
    pickle.dump(new_data, f)