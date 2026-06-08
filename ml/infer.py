#!/usr/bin/env python3
"""
Run inference on a features CSV using a pre-trained model.

Loads features_data.csv produced by accel_receiver.py, predicts the class
for each window, and writes results.csv with ground truth vs prediction.

Usage:
    python3 infer.py [--data features_data.csv] [--model model.pkl] [--out results.csv]
"""

import argparse
import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLS = [
    'rms_x',  'rms_y',  'rms_z',
    'peak_x', 'peak_y', 'peak_z',
    'std_x',  'std_y',  'std_z',
    'p2p_x',  'p2p_y',  'p2p_z',
]
LABEL_COL = 'label'


def main():
    parser = argparse.ArgumentParser(description='Run inference on features CSV')
    parser.add_argument('--data',  default='features_data.csv',
                        help='Input features CSV (default: features_data.csv)')
    parser.add_argument('--model', default='model.pkl',
                        help='Trained model file (default: model.pkl)')
    parser.add_argument('--out',   default='results.csv',
                        help='Output results CSV (default: results.csv)')
    args = parser.parse_args()

    model = joblib.load(args.model)
    print(f'Model loaded from {args.model}')

    df = pd.read_csv(args.data)
    print(f'Loaded {len(df)} samples from {args.data}\n')

    X = df[FEATURE_COLS]
    y_true = df[LABEL_COL]

    y_pred = model.predict(X)

    df['prediction'] = y_pred
    df['correct']    = y_true == y_pred

    results = df[['seq', 't_ms', 'recv_ms', LABEL_COL, 'prediction', 'correct']]
    results.to_csv(args.out, index=False)
    print(f'Results written to {args.out}\n')

    accuracy = df['correct'].mean()
    print(f'Accuracy: {accuracy:.4f}  ({df["correct"].sum()}/{len(df)} correct)\n')

    print('Classification report:')
    print(classification_report(y_true, y_pred))

    print('Confusion matrix:')
    labels = sorted(y_true.unique())
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = f'{"":>12}' + ''.join(f'{l:>12}' for l in labels)
    print(header)
    for label, row in zip(labels, cm):
        print(f'{label:>12}' + ''.join(f'{v:>12}' for v in row))


if __name__ == '__main__':
    main()
