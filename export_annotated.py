import argparse
import pandas as pd
import glob
import os
import sys

# try to pick latest labeled file automatically unless overridden
def find_latest_labeled():
    cands = sorted(glob.glob("*labeled*.csv") + glob.glob("ml_insurance_challenge_*.csv"),
                   key=os.path.getmtime, reverse=True)
    return cands[0] if cands else None

def find_label_column(df):
    candidates = ['insurance_labels', 'insurance_label', 'labels', 'label', 'predicted_labels']
    for c in candidates:
        if c in df.columns:
            return c
    return None

def find_name_column(df):
    candidates = ['company_name', 'name', 'website', 'url', 'company']
    for c in candidates:
        if c in df.columns:
            return c
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled", help="Path to labeled CSV (optional). If omitted, script auto-detects latest labeled file.")
    parser.add_argument("--orig", default="ml_insurance_challenge_cleaned.csv", help="Original input CSV")
    parser.add_argument("--out", default="ml_insurance_challenge_annotated.csv", help="Output annotated CSV")
    args = parser.parse_args()

    labeled_path = args.labeled or find_latest_labeled()
    if not labeled_path:
        print("No labeled CSV found. Provide --labeled <path> or place a labeled file matching *labeled*.csv in the folder.")
        sys.exit(1)

    if not os.path.exists(args.orig):
        print("Original input file not found:", args.orig)
        sys.exit(1)

    print("Using labeled file:", labeled_path)
    orig = pd.read_csv(args.orig, dtype=str).fillna('')
    labeled = pd.read_csv(labeled_path, dtype=str).fillna('')

    label_col = find_label_column(labeled)
    if 'id' in orig.columns and 'id' in labeled.columns:
        merged = orig.merge(labeled[['id', label_col]] if label_col else labeled[['id']],
                            on='id', how='left')
        # normalize column name
        if label_col:
            merged = merged.rename(columns={label_col: 'insurance_label'})
        else:
            # if no label col, try to find any column that looks like labels
            print("WARNING: labeled file contains no recognized label column. Available columns:", list(labeled.columns))
            sys.exit(1)
    else:
        # try positional merge if lengths match
        if len(orig) == len(labeled):
            if label_col:
                orig['insurance_label'] = labeled[label_col].astype(str).fillna('')
            else:
                print("No recognized label column in labeled file. Available columns:", list(labeled.columns))
                sys.exit(1)
            merged = orig
        else:
            # try merging by company name column
            name_orig = find_name_column(orig)
            name_lab = find_name_column(labeled)
            if name_orig and name_lab:
                merged = orig.merge(labeled[[name_lab, label_col]] if label_col else labeled[[name_lab]],
                                    left_on=name_orig, right_on=name_lab, how='left')
                if label_col:
                    merged = merged.rename(columns={label_col: 'insurance_label'})
                else:
                    print("No recognized label column in labeled file. Available columns:", list(labeled.columns))
                    sys.exit(1)
            else:
                print("Cannot align original and labeled files automatically.")
                print("Original columns:", list(orig.columns))
                print("Labeled columns:", list(labeled.columns))
                print("Provide a labeled file with an 'id' column matching original, or pass --labeled for a file with same row-order.")
                sys.exit(1)

    merged.to_csv(args.out, index=False)
    print("Saved annotated:", args.out)
    print("Source labeled file used:", labeled_path)

if __name__ == "__main__":
    main()