#!/usr/bin/env python3
"""
Train a quantized TFLite MLP classifier on features_data.csv.
Produces model.tflite (INT8 internal ops, float32 I/O) for deployment
on iMX8MP — runs on CPU (exp-a) or NPU via VX delegate (exp-b).

Usage:
    python3 train.py [--data features_data.csv] [--out model.tflite]
"""

import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLS = [
    'rms_x',  'rms_y',  'rms_z',
    'peak_x', 'peak_y', 'peak_z',
    'std_x',  'std_y',  'std_z',
    'p2p_x',  'p2p_y',  'p2p_z',
]
LABEL_COL  = 'label'
NUM_CLASSES = 3
EPOCHS      = 50
BATCH_SIZE  = 16


def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ])


def main():
    parser = argparse.ArgumentParser(description='Train quantized TFLite MLP')
    parser.add_argument('--data', default='features_data.csv',
                        help='Features CSV produced by accel_receiver.py (default: features_data.csv)')
    parser.add_argument('--out',  default='model.tflite',
                        help='Output TFLite model path (default: model.tflite)')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f'Loaded {len(df)} samples from {args.data}')
    print('Class distribution:')
    print(df[LABEL_COL].value_counts().to_string())
    print()

    le = LabelEncoder()
    X  = df[FEATURE_COLS].values.astype(np.float32)
    y  = le.fit_transform(df[LABEL_COL])

    # Save class order so inference scripts can map indices back to names
    labels_path = args.out.replace('.tflite', '_labels.txt')
    with open(labels_path, 'w') as f:
        f.write('\n'.join(le.classes_))
    print(f'Class order: {list(le.classes_)}  (saved to {labels_path})\n')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_model()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.fit(X_train, y_train,
              epochs=EPOCHS,
              batch_size=BATCH_SIZE,
              validation_split=0.1,
              verbose=1)

    _, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print(f'\nTest accuracy: {acc:.4f}')
    print('\nClassification report:')
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print('Confusion matrix:')
    cm = confusion_matrix(y_test, y_pred)
    header = f'{"":>12}' + ''.join(f'{c:>12}' for c in le.classes_)
    print(header)
    for label, row in zip(le.classes_, cm):
        print(f'{label:>12}' + ''.join(f'{v:>12}' for v in row))

    # Convert to INT8-quantized TFLite (required for NPU via VX delegate)
    def representative_dataset():
        for sample in X_train:
            yield [sample.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations                 = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset        = representative_dataset
    converter.target_spec.supported_ops     = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Float32 I/O keeps the Python interface simple; internal ops are INT8
    converter.inference_input_type          = tf.float32
    converter.inference_output_type         = tf.float32

    tflite_model = converter.convert()
    with open(args.out, 'wb') as f:
        f.write(tflite_model)
    print(f'\nQuantized TFLite model saved to {args.out}  ({len(tflite_model):,} bytes)')


if __name__ == '__main__':
    main()
