# ----------------------------------------------------------------------------------------------------
# This file is based on: totalseg_get_phase.py (GitHub: wasserth/TotalSegmentator)
# Original author(s): Jakob Wasserthal
# Modified by: Benjamin Hou on Apr 2, 2025
# Description: Adapted XGBoost Classification for CT Contrast Phase Prediction
# ----------------------------------------------------------------------------------------------------

import sys
from pathlib import Path
import time
import argparse
import json
import pickle
from pprint import pprint
import importlib.resources
import importlib.metadata

import nibabel as nib
import numpy as np

from totalsegmentator.python_api import totalsegmentator


phase = {
    0: "noncontrast",
    1: "arterial",
    2: "venous",
    3: "delayed",
}

def get_ct_contrast_phase(ct_img: nib.Nifti1Image, model_file: Path = None):

    organs = ["liver", "pancreas", "urinary_bladder", "gallbladder",
              "heart", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
              "iliac_vena_left", "iliac_vena_right", "iliac_artery_left", "iliac_artery_right",
              "pulmonary_vein", "brain", "colon", "small_bowel"]

    st = time.time()
    seg_img, stats = totalsegmentator(ct_img, None, ml=True, fast=True, statistics=True, 
                                      roi_subset=None, statistics_exclude_masks_at_border=False,
                                      quiet=True, stats_aggregation="median")
    # print(f"ts took: {time.time()-st:.2f}s")

    features = [stats[organ]["intensity"] for organ in organs]
    features = [np.nan if x == 0.0 else x for x in features]

    all_models = pickle.load(open(model_file, "rb"))

    # ensemble across folds
    logits = [model["model"].predict_proba([features])[0] for model in all_models]
    logits = np.array(logits)
    logits = np.mean(logits, axis=0)
    y_pred = np.argmax(logits)

    return {"phase": phase[y_pred], "class_id": str(y_pred), "logits": logits.tolist()}


def main():
    """
    Predicts the contrast phase of a CT scan. Specifically this script will predict the
    phase class of a CT scan based on the intensity of different regions in the image.
    """
    parser = argparse.ArgumentParser(description="Get CT contrast phase.")

    parser.add_argument("-i", metavar="filepath", dest="input_file",
                        help="path to CT file",
                        type=lambda p: Path(p).absolute(), required=True)

    parser.add_argument("-o", metavar="filepath", dest="output_file",
                        help="path to output json file",
                        type=lambda p: Path(p).absolute(), required=True)
    
    parser.add_argument("-m", metavar="filepath", dest="model_file",
                        help="path to classifier model",
                        type=lambda p: Path(p).absolute(), required=False, default=None)
    
    parser.add_argument("-q", dest="quiet", action="store_true",
                        help="Print no output to stdout", default=False)

    args = parser.parse_args()

    res = get_ct_contrast_phase(nib.load(args.input_file), args.model_file)

    if not args.quiet:
        print("Result:")
        pprint(res)

    with open(args.output_file, "w") as f:
        f.write(json.dumps(res, indent=4))

if __name__ == "__main__":
    main()
