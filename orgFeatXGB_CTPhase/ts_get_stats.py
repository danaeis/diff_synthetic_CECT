import pickle
import nibabel as nib


from totalsegmentator.python_api import totalsegmentator


def main(input_path, output_path):

    ct_img = nib.load(input_path)

    seg_img, stats = totalsegmentator(ct_img, None, ml=True, fast=True, statistics=True,
                                   roi_subset=None, statistics_exclude_masks_at_border=False,
                                   quiet=False, stats_aggregation="median")

    with open(output_path, 'wb') as f:
        pickle.dump(stats, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process a NIfTI file and save output to a pickle.")
    parser.add_argument("input_path", help="Path to the input NIfTI file (.nii or .nii.gz)")
    parser.add_argument("output_path", help="Path to save the output pickle file")

    args = parser.parse_args()
    main(args.input_path, args.output_path)
