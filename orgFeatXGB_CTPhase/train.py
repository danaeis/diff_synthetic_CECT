import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report
from tqdm import tqdm

data = 'vindr_ds'
data_root = '/media/disk1/saeedeh_danaei/ncct_cect/vindr_ds/'

organs = ["liver", "pancreas", "urinary_bladder", "gallbladder",
          "heart", "aorta", "inferior_vena_cava", "portal_vein_and_splenic_vein",
          "iliac_vena_left", "iliac_vena_right", "iliac_artery_left", "iliac_artery_right",
          "pulmonary_vein", "brain", "colon", "small_bowel"]

# Load metadata
# metadata_df = pd.read_csv(os.path.join(data_root, 'ct_hcc_metadata_v2.csv'))
metadata_df = pd.read_csv(os.path.join(data_root, 'vindr_nifti_metadata.csv'))
# metadata_df['filepath'] = metadata_df.apply(
#     lambda x: os.path.join(data_root, 'stats', x['ct_file_name']) + '_scan.pkl', axis=1)


# Initialize GroupKFold
sgkf = StratifiedGroupKFold(n_splits=5, random_state=42, shuffle=True)

# Convert the split generator to a list and access the desired split directly
patient = 'patient'
series = 'series'

if data == "waw_tace":
    patient = 'patient_id'
    series = 'ct_file_name'
elif data == "vindr_ds":
    patient = 'StudyInstanceUID'
    series = 'SeriesInstanceUID'
    
splits = sgkf.split(X=metadata_df[series],
                    y=metadata_df['ct_phase'],
                    groups=metadata_df[patient])
splits = list(splits)


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

all_models = []
for fold in range(5):

    print(f"\n🧪 Training fold {fold+1}/5")

    # Get train and test indices from the desired split
    train_indices, test_indices = splits[fold]

    train_df = metadata_df.iloc[train_indices]
    test_df = metadata_df.iloc[test_indices]

    # Load all train data
    X_train=[]; y_train=[]
    print("metadata columns", metadata_df.columns)
    for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
        stats = load_pickle(row['stats_path'])

        features = []
        for organ in organs:
            features.append(stats[organ]["intensity"])
        features = [np.nan if x == 0.0 else x for x in features]

        X_train.append(features)
        y_train.append(row['ct_phase'])

    # Load all test data
    X_test=[]; y_test=[]
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        stats = load_pickle(row['stats_path'])

        features = []
        for organ in organs:
            features.append(stats[organ]["intensity"])
        features = [np.nan if x == 0.0 else x for x in features]

        X_test.append(features)
        y_test.append(row['ct_phase'])
    # Convert to DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Define and train model
    model = xgb.XGBClassifier(
        learning_rate=0.05,
        max_depth=4,
        n_estimators=200,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='mlogloss' if len(np.unique(y_train)) > 2 else 'logloss'
    )
    
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    # Save to a text file
    output_file = "classification_report.txt"
    with open(output_file, 'a') as f:
        f.write(f"=== Fold {fold+1} ===\n")
        f.write(report)
        f.write("\n\n")
    print(report, f"saved to {output_file}")

    # Save model in the list
    all_models.append({
        "fold": fold,
        "model": model,
    })

import re

# Path to your saved classification report
report_path = "classification_report.txt"

# Storage for each fold's metrics
accuracies = []
macro_precisions = []
macro_recalls = []
macro_f1s = []

with open(report_path, 'r') as f:
    lines = f.readlines()

# Go through each line and extract metrics
for line in lines:
    if "accuracy" in line:
        accuracy = float(re.findall(r"\d+\.\d+", line)[0])
        accuracies.append(accuracy)
    elif "macro avg" in line:
        parts = re.findall(r"\d+\.\d+", line)
        precision, recall, f1 = map(float, parts[:3])
        macro_precisions.append(precision)
        macro_recalls.append(recall)
        macro_f1s.append(f1)

# Compute averages
def avg(lst): return sum(lst) / len(lst)

final_report = {
    "Accuracy": avg(accuracies),
    "Macro Precision": avg(macro_precisions),
    "Macro Recall": avg(macro_recalls),
    "Macro F1 Score": avg(macro_f1s)
}

# Print nicely
print("\n📊 Final Averaged Classification Report (Across Folds):")
for metric, value in final_report.items():
    print(f"{metric:<20}: {value:.4f}")



# Save all models in one file
if data == "vindr_ds":
    with open("xgb_vindr.pkl", "wb") as f:
        pickle.dump(all_models, f)
elif data == "waw_tace":
    with open("xgb_wawtace.pkl", "wb") as f:
        pickle.dump(all_models, f)
print("✅ All fold models saved to .pkl")
