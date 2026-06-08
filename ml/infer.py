#!/usr/bin/env python3
"""
Run offline batch inference on a features CSV using a quantized TFLite model.

Loads features_data.csv produced by accel_receiver.py, predicts the class
for each window, and writes results.csv with ground truth vs prediction.

Usage:
    python3 infer.py [--data features_data.csv] [--model model.tflite] [--out results.csv]
    python3 infer.py --ext_delegate /usr/lib/libvx_delegate.so  # NPU (exp-b)
"""

import argparse
import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
from sklearn.metrics import classification_report, confusion_matrix

FEATURE_COLS = [
    'rms_x',  'rms_y',  'rms_z',
    'peak_x', 'peak_y', 'peak_z',
    'std_x',  'std_y',  'std_z',
    'p2p_x',  'p2p_y',  'p2p_z',
]
LABEL_COL = 'label'


def load_interpreter(model_path, ext_delegate=None, ext_delegate_options=None, num_threads=None):
    delegates = None
    if ext_delegate:
        opts = ext_delegate_options or {}
        print(f'Loading delegate from {ext_delegate} with opts: {opts}')
        delegates = [tflite.load_delegate(ext_delegate, opts)]

    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=delegates,
        num_threads=num_threads,
    )
    interpreter.allocate_tensors()

    # Warm-up invoke — first call triggers JIT compilation on NPU
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    dummy = np.zeros((1, len(FEATURE_COLS)), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()

    return interpreter, input_details, output_details


def predict_batch(interpreter, input_details, output_details, X):
    preds = []
    for sample in X:
        interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, -1).astype(np.float32))
        interpreter.invoke()
        out = interpreter.get_tensor(output_details[0]['index'])
        preds.append(int(np.argmax(np.squeeze(out))))
    return np.array(preds)


def main():
    parser = argparse.ArgumentParser(description='Offline TFLite inference on features CSV')
    parser.add_argument('--data',    default='features_data.csv',
                        help='Input features CSV (default: features_data.csv)')
    parser.add_argument('--model',   default='model.tflite',
                        help='TFLite model file (default: model.tflite)')
    parser.add_argument('--labels',  default='model_labels.npy',
                        help='Label mapping from train.py (default: model_labels.npy)')
    parser.add_argument('--out',     default='results.csv',
                        help='Output results CSV (default: results.csv)')
    parser.add_argument('-e', '--ext_delegate',
                        help='External delegate library path (e.g. /usr/lib/libvx_delegate.so)')
    parser.add_argument('-o', '--ext_delegate_options',
                        help='Delegate options, format: "key1: val1; key2: val2"')
    parser.add_argument('--num_threads', type=int, default=None,
                        help='Number of threads for CPU inference')
    args = parser.parse_args()

    delegate_opts = {}
    if args.ext_delegate_options:
        for item in args.ext_delegate_options.split(';'):
            kv = item.split(':')
            if len(kv) == 2:
                delegate_opts[kv[0].strip()] = kv[1].strip()

    interpreter, input_details, output_details = load_interpreter(
        args.model, args.ext_delegate, delegate_opts, args.num_threads)
    print(f'Model loaded from {args.model}')

    le_classes = np.load(args.labels, allow_pickle=True)
    print(f'Classes: {list(le_classes)}\n')

    df = pd.read_csv(args.data)
    print(f'Loaded {len(df)} samples from {args.data}\n')

    X      = df[FEATURE_COLS].values
    y_true = df[LABEL_COL].values

    y_idx  = predict_batch(interpreter, input_details, output_details, X)
    y_pred = le_classes[y_idx]

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
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    header = f'{"":>12}' + ''.join(f'{l:>12}' for l in labels)
    print(header)
    for label, row in zip(labels, cm):
        print(f'{label:>12}' + ''.join(f'{v:>12}' for v in row))


if __name__ == '__main__':
    main()
