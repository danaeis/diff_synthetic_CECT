# CT Contrast Phase Classification with TS and XGBoost

This model implements an XGBoost-based classifier to automatically predict the contrast phase of abdominal CT scans. The model is trained on WAW-TACE dataset, and classifies scans into key contrast phases: **non-contrast**, **arterial**, **venous**, and **delayed**.

## ⚙️ Python Environment

```text
Installation Info:
- Date installed: Mar 6, 2025.
- Python version: 3.9.15
- TotalSegmentator version: 2.7.0 
```

For full list of package versions, see [requirements.txt](requirements.txt).

## 🔍 Run Inference

To predict the phase of new CT scans run `totalseg_get_phase.py`. Latest model is currently: `xgb_wawtace.pkl`.

```text
[user@machine ~]$ python totalseg_get_phase.py --help
usage: totalseg_get_phase.py [-h] -i filepath -o filepath [-m filepath] [-q]

Get CT contrast phase.

optional arguments:
  -h, --help   show this help message and exit
  -i filepath  path to CT file
  -o filepath  path to output json file
  -m filepath  path to classifier model
  -q           Print no output to stdout
```

Example output:
```json
{
    "phase": "delayed",
    "class_id": "3",
    "logits": [
        0.0004899133346043527,
        0.0005343952216207981,
        0.005297648720443249,
        0.993678092956543
    ]
}
```

## 🧪 Replicating Experiment

<details>

<summary>Details</summary>

1. **Download WAW-TACE Dataset**  
   Download the dataset from [Zenodo](https://zenodo.org/records/12741586).  
   - Ensure that `ct_hcc_metadata_v2.csv` is placed in the project root directory.  
   - Extract the contents of `ct_scans_[1-4]_4_wawtace_09_05_24.zip` into the `images/` folder.

2. **Extract Radiomic Statistics with TotalSegmentator (TS)**  
   Use TotalSegmentator to segment abdominal organs and extract radiomic features. The features are saved as `.pkl` files.

   **Example (single case):**
   ```bash
   python ts_get_stats.py /path/to/WAW-TACE/images/388/388_2_scan.nii.gz stats/388_2_scan.pkl
   ```

   **Batch processing (parallelized):**  
   To process multiple scans in parallel, use GNU `parallel`:
   ```bash
   parallel --jobs 4 < ts_get_stats.sh
   ```
   
   Make sure `ts_get_stats.sh` contains one command per line.

3. **Train the XGBoost Model**  
   Set the `data_root` variable in `train.py` to the directory containing the dumped `.pkl` stats files, then run:

   ```bash
   python train.py
   ```
   
</details>

## 🙏 Acknowledgement

This research was supported by the Intramural Research Program of the National Institutes of Health (NIH); National Library of Medicine (NLM) and Clinical Center (CC). This work utilized the computational resources of the NIH high-performance computing Biowulf cluster ([https://hpc.nih.gov/](https://hpc.nih.gov/)).

## 📖 Citation

If you use this code or model in your research, please cite:

```
@article{hou2025segment,
  title={Segment-and-Classify: ROI-Guided Generalizable Contrast Phase Classification in CT Using XGBoost},
  author={Hou, Benjamin and Mathai, Tejas Sudharshan and Mukherjee, Pritam and Wang, Xinya and Summers, Ronald M and Lu, Zhiyong},
  journal={arXiv preprint arXiv:2501.14066},
  year={2025}
}

@article{wasserthal2023totalsegmentator,
  title={TotalSegmentator: robust segmentation of 104 anatomic structures in CT images},
  author={Wasserthal, Jakob and Breit, Hanns-Christian and Meyer, Manfred T and Pradella, Maurice and Hinck, Daniel and Sauter, Alexander W and Heye, Tobias and Boll, Daniel T and Cyriac, Joshy and Yang, Shan and others},
  journal={Radiology: Artificial Intelligence},
  volume={5},
  number={5},
  pages={e230024},
  year={2023},
  publisher={Radiological Society of North America}
}
```
