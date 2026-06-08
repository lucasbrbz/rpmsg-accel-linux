#!/usr/bin/env python3
"""
Experiment A — Linux CPU baseline (Jetson simulation).

Receives raw FRAME_RAW_ACCEL samples from the M7, accumulates them into
windows of WINDOW_SIZE, computes features on the A53 CPU, runs TFLite
inference on CPU, and writes features_data.csv.

Usage:
    python3 accel_receiver.py --model ../ml/model.tflite
"""

import argparse
import csv
import math
import os
import signal
import struct
import sys
import termios
import time
import tty

import numpy as np
import psutil
import tflite_runtime.interpreter as tflite

RPMSG_DEVICE  = '/dev/ttyRPMSG0'
FRAME_MAGIC   = 0xA55AA55A
FRAME_VERSION = 1
WINDOW_SIZE   = 20

MAGIC_LE = struct.pack('<I', FRAME_MAGIC)

FRAME_HEADER_FORMAT = '<IBBIIBBH'
FRAME_HEADER_SIZE   = struct.calcsize(FRAME_HEADER_FORMAT)   # 18 bytes

RAW_ACCEL_PAYLOAD_FORMAT = '<hhh'
RAW_ACCEL_PAYLOAD_SIZE   = struct.calcsize(RAW_ACCEL_PAYLOAD_FORMAT)  # 6 bytes

FRAME_RAW_ACCEL = 1

CSV_HEADER = ['seq', 't_ms', 'recv_ms', 'label', 'prediction',
              'rms_x',  'rms_y',  'rms_z',
              'peak_x', 'peak_y', 'peak_z',
              'std_x',  'std_y',  'std_z',
              'p2p_x',  'p2p_y',  'p2p_z',
              't_feat_ms', 't_infer_ms', 't_window_ms',
              'window_interval_ms', 'cpu_percent', 'mem_rss_kb']

LABELS = {0: 'UNKNOWN', 1: 'NORMAL', 2: 'IMBALANCE', 3: 'ANOMALY'}

fd       = None
csv_file = None


def cleanup(signum, frame):
    print('\nInterrupted. Closing device.')
    if fd is not None:
        os.close(fd)
    if csv_file is not None:
        csv_file.close()
        print('Dataset saved.')
    sys.exit(0)


def read_exact(fd, n):
    buf = b''
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if chunk:
            buf += chunk
    return buf


def read_frame(fd):
    skipped = 0
    window = b''
    while True:
        window += os.read(fd, 1)
        if len(window) < 4:
            continue
        if len(window) > 4:
            window = window[-4:]
        if window == MAGIC_LE:
            break
        skipped += 1

    hdr_rest    = read_exact(fd, FRAME_HEADER_SIZE - 4)
    header_raw  = MAGIC_LE + hdr_rest
    fields      = struct.unpack(FRAME_HEADER_FORMAT, header_raw)
    payload_len = fields[7]
    payload_raw = read_exact(fd, payload_len)
    return fields, payload_raw, skipped


def compute_features(window):
    xs = [s[0] for s in window]
    ys = [s[1] for s in window]
    zs = [s[2] for s in window]

    def stats(vals):
        n    = len(vals)
        mean = sum(vals) / n
        rms  = math.sqrt(sum(v * v for v in vals) / n)
        peak = max(abs(v) for v in vals)
        std  = math.sqrt(sum((v - mean) ** 2 for v in vals) / n)
        p2p  = max(vals) - min(vals)
        return rms, peak, std, p2p

    rx, px, sx, ppx = stats(xs)
    ry, py, sy, ppy = stats(ys)
    rz, pz, sz, ppz = stats(zs)

    return rx, ry, rz, px, py, pz, sx, sy, sz, ppx, ppy, ppz


def load_interpreter(model_path, num_threads=None):
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()

    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy = np.zeros((1, 12), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], dummy)
    interpreter.invoke()

    return interpreter, input_details, output_details


def predict(interpreter, input_details, output_details, le_classes, features):
    x = np.array(features, dtype=np.float32).reshape(1, -1)
    interpreter.set_tensor(input_details[0]['index'], x)
    t0 = time.perf_counter()
    interpreter.invoke()
    t_infer_ms = (time.perf_counter() - t0) * 1000.0
    out = interpreter.get_tensor(output_details[0]['index'])
    return le_classes[int(np.argmax(np.squeeze(out)))], t_infer_ms


def main():
    global fd, csv_file

    parser = argparse.ArgumentParser(description='RPMsg raw accel receiver — exp-a (CPU inference)')
    parser.add_argument('--device', default=RPMSG_DEVICE,
                        help=f'RPMsg tty device (default: {RPMSG_DEVICE})')
    parser.add_argument('--model',  default='../ml/model.tflite',
                        help='TFLite model path (default: ../ml/model.tflite)')
    parser.add_argument('--labels', default='../ml/model_labels.npy',
                        help='Label mapping from train.py (default: ../ml/model_labels.npy)')
    parser.add_argument('--num_threads', type=int, default=None,
                        help='Number of CPU threads for inference')
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    interpreter, input_details, output_details = load_interpreter(
        args.model, args.num_threads)
    le_classes = np.load(args.labels, allow_pickle=True)
    print(f'Model ready (CPU). Classes: {list(le_classes)}')

    proc = psutil.Process()
    proc.cpu_percent(interval=None)   # prime the counter — first call always returns 0.0

    csv_path = 'features_data.csv'
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)
    print(f'Writing dataset to {csv_path}  (window={WINDOW_SIZE} samples)')

    print(f'Opening {args.device}...')
    fd = os.open(args.device, os.O_RDWR)
    tty.setraw(fd)
    termios.tcflush(fd, termios.TCIFLUSH)

    os.write(fd, b'\x00')
    print('Handshake sent. Waiting for data...\n')
    print(f'{"seq":>8}  {"t_ms":>9}  {"label":<12}  {"pred":<12}  '
          f'{"t_feat":>8}  {"t_infer":>8}  {"cpu%":>6}')
    print('-' * 88)

    sample_buf    = []
    window_seq    = 0
    window_t_ms   = 0
    window_recv   = 0.0
    window_label  = ''
    prev_win_time = None   # perf_counter at last window completion

    while True:
        (magic, version, ftype, seq, ts_ms, label, flags, payload_len), payload_raw, skipped = \
            read_frame(fd)
        recv_ms = time.monotonic() * 1000.0

        if skipped:
            print(f'[warn] resync: skipped {skipped} byte(s)', file=sys.stderr)
        if version != FRAME_VERSION:
            print(f'[warn] unknown version {version}', file=sys.stderr)
            continue
        if ftype != FRAME_RAW_ACCEL:
            print(f'[warn] unexpected frame type {ftype}, skipping', file=sys.stderr)
            continue

        x, y, z   = struct.unpack(RAW_ACCEL_PAYLOAD_FORMAT, payload_raw)
        label_str = LABELS.get(label, f'UNKNOWN({label})')

        if len(sample_buf) == 0:
            window_t_ms  = ts_ms
            window_recv  = recv_ms

        sample_buf.append((x, y, z))
        window_label = label_str

        if len(sample_buf) >= WINDOW_SIZE:
            t0_feat = time.perf_counter()
            feats   = compute_features(sample_buf)
            t_feat_ms = (time.perf_counter() - t0_feat) * 1000.0

            rms_x,  rms_y,  rms_z, \
            peak_x, peak_y, peak_z, \
            std_x,  std_y,  std_z, \
            p2p_x,  p2p_y,  p2p_z  = feats

            prediction, t_infer_ms = predict(
                interpreter, input_details, output_details, le_classes, list(feats))

            now = time.perf_counter()
            t_window_ms        = t_feat_ms + t_infer_ms
            window_interval_ms = (now - prev_win_time) * 1000.0 if prev_win_time else 0.0
            prev_win_time      = now

            cpu_percent = proc.cpu_percent(interval=None)
            mem_rss_kb  = proc.memory_info().rss // 1024

            match = 'OK' if prediction == window_label else 'MISMATCH'

            writer.writerow([window_seq, window_t_ms, f'{window_recv:.3f}',
                             window_label, prediction,
                             f'{rms_x:.3f}',  f'{rms_y:.3f}',  f'{rms_z:.3f}',
                             f'{peak_x:.3f}', f'{peak_y:.3f}', f'{peak_z:.3f}',
                             f'{std_x:.3f}',  f'{std_y:.3f}',  f'{std_z:.3f}',
                             f'{p2p_x:.3f}',  f'{p2p_y:.3f}',  f'{p2p_z:.3f}',
                             f'{t_feat_ms:.3f}', f'{t_infer_ms:.3f}', f'{t_window_ms:.3f}',
                             f'{window_interval_ms:.3f}', f'{cpu_percent:.1f}', mem_rss_kb])
            csv_file.flush()

            print(f'  >> win {window_seq:>4}  [{window_label}] → {prediction}  [{match}]  '
                  f'feat={t_feat_ms:.2f}ms  infer={t_infer_ms:.2f}ms  '
                  f'cpu={cpu_percent:.1f}%  rss={mem_rss_kb}KB')

            sample_buf  = []
            window_seq += 1


if __name__ == '__main__':
    main()
