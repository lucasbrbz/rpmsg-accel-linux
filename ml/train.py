#!/usr/bin/env python3
"""
Train a Random Forest classifier on features_data.csv and serialize the model.

Usage:
    python3 train.py [--data features_data.csv] [--out model.pkl]
"""

import argparse
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLS = [
    'rms_x',  'rms_y',  'rms_z',
    'peak_x', 'peak_y', 'peak_z',
    'std_x',  'std_y',  'std_z',
    'p2p_x',  'p2p_y',  'p2p_z',
]
LABEL_COL = 'label'


def main():
    parser = argparse.ArgumentParser(description='Train RF classifier on feature data')
    parser.add_argument('--data', default='features_data.csv',
                        help='Path to the features CSV (default: features_data.csv)')
    parser.add_argument('--out', default='model.pkl',
                        help='Output model path (default: model.pkl)')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f'Loaded {len(df)} samples from {args.data}')
    print('Class distribution:')
    print(df[LABEL_COL].value_counts().to_string())
    print()

    X = df[FEATURE_COLS]
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f'Test accuracy: {(y_pred == y_test).mean():.4f}')
    print()
    print('Classification report:')
    print(classification_report(y_test, y_pred))
    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred, labels=model.classes_))
    print()

    print('Feature importances:')
    for col, imp in sorted(zip(FEATURE_COLS, model.feature_importances_),
                           key=lambda x: x[1], reverse=True):
        print(f'  {col:<10} {imp:.4f}')

    joblib.dump(model, args.out)
    print(f'\nModel saved to {args.out}')


if __name__ == '__main__':
    main()
