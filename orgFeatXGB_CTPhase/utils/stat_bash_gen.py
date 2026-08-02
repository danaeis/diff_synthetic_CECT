
import pandas as pd
import os

df = pd.read_csv("vindr_nifti_metadata.csv")

stats_paths = []
with open("vindr_ts_get_stats.sh", "w") as f:
    for _, row in df.iterrows():
        nifti_path = row["orig_volume_path"]
        # Replace .nii.gz or .nii with .pkl in the same path
        if nifti_path.endswith('.nii.gz'):
            pkl_path = nifti_path[:-7] + '.pkl'
        elif nifti_path.endswith('.nii'):
            pkl_path = nifti_path[:-4] + '.pkl'
        else:
            raise ValueError(f"Unknown NIfTI extension in path: {nifti_path}")
        f.write(f"python ts_get_stats.py {nifti_path} {pkl_path}\n")
        stats_paths.append(pkl_path)
# Add stats_path column and save updated CSV

df["stats_path"] = stats_paths
df.to_csv("vindr_nifti_metadata.csv", index=False)



