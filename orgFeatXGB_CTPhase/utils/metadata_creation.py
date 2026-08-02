from numpy import column_stack
import pandas as pd

# Set your root path here (include trailing slash if needed)
root_path = "../../ncct_cect/vindr_ds/"  # <-- CHANGE THIS

# Load the CSV
df = pd.read_csv("vindr_nifti_metadata.csv")

# Add the new column
df["orig_volume_path"] = (
    root_path + "original_volumes/" + df["StudyInstanceUID"].astype(str) + "/" + df["SeriesInstanceUID"].astype(str) + ".nii.gz"
)

# Define your mapping
ct_phase = {
    "non-contrast": 0,
    "aterial": 1,
    "venous": 2,
    "unknown": -1
}

# Map the Phase column to numbers
df["ct_phase"] = df["Phase"].map(ct_phase)

# Save the updated CSV (overwrite or use a new file)
df.to_csv("vindr_nifti_metadata.csv", index=False)