import pickle
import pandas as pd
import numpy as np

# Path to your pickle file
pickle_file_path = '../../ncct_cect/vindr_ds/original_volumes/1.2.840.113619.2.278.3.717616.310.1589696975.113/1.2.840.113619.2.278.3.717616.310.1589696975.214.4202498.pkl'

# Load the pickle file
with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

# Inspect the contents
print("Type of loaded data:", type(data))

# If it's a dictionary, print keys
if isinstance(data, dict):
    print("Keys in dictionary:", list(data.keys()))

# If it's a DataFrame, display basic info
if isinstance(data, pd.DataFrame):
    print("\nDataFrame Info:")
    print(data.info())
    print("\nFirst few rows:")
    print(data.head())

# If it's a list or array, print length and sample
if isinstance(data, (list, np.ndarray)):
    print("Length of data:", len(data))
    print("Sample of data:", data[:5])

# General summary
print("\nData preview:", data)