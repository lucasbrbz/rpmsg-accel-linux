#!/usr/bin/env python3
"""
Experiment B — iMX8MP native workload partitioning.

Receives FRAME_FEATURES frames from the M7 core via RPMsg, runs TFLite
inference (NPU via VX delegate or CPU fallback), and writes features_data.csv.

Feature extraction happens on the M7, so t_feat_ms is logged as 0.0 —
the A53's workload is receive + inference only.

Usage (NPU via VX delegate):
    python3 accel_receiver.py --model ../ml/model.tflite \
        --ext_delegate /usr/lib/libvx_delegate.so

Usage (CPU fallback):
    python3 accel_receiver.py --model ../ml/model.tflite
"""

import argparse
import csv
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

MAGIC_LE = struct.pack('<I', FRAME_MAGIC)

FRAME_HEADER_FORMAT = '<IBBIIBBH'
FRAME_HEADER_SIZE   = struct.calcsize(FRAME_HEADER_FORMAT)   # 18 bytes

FEATURES_PAYLOAD_FORMAT = '<' + 'f' * 12 + 'HH'
FEATURES_PAYLOAD_SIZE   = struct.calcsize(FEATURES_PAYLOAD_FORMAT)  # 52 bytes

FRAME_RAW_ACCEL = 1
FRAME_FEATURES  = 2
FRAME_STATUS    = 3

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


def load_interpreter(model_path, ext_delegate=None, ext_delegate_options=None, num_threads=None):
    delegates = None
    if ext_delegate:
        opts = ext_delegate_options or {}
        print(f'Loading NPU delegate from {ext_delegate}')
        delegates = [tflite.load_delegate(ext_delegate, opts)]

    interpreter = tflite.Interpreter(
        model_path=model_path,
        experimental_delegates=delegates,
        num_threads=num_threads,
    )
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

    parser = argparse.ArgumentParser(description='RPMsg feature frame receiver — exp-b (NPU inference)')
    parser.add_argument('--device', default=RPMSG_DEVICE,
                        help=f'RPMsg tty device (default: {RPMSG_DEVICE})')
    parser.add_argument('--model',  default='../ml/model.tflite',
                        help='TFLite model path (default: ../ml/model.tflite)')
    parser.add_argument('--labels', default='../ml/model_labels.npy',
                        help='Label mapping from train.py (default: ../ml/model_labels.npy)')
    parser.add_argument('-e', '--ext_delegate',
                        help='External delegate library (e.g. /usr/lib/libvx_delegate.so)')
    parser.add_argument('-o', '--ext_delegate_options',
                        help='Delegate options, format: "key1: val1; key2: val2"')
    parser.add_argument('--num_threads', type=int, default=None,
                        help='Number of CPU threads (no delegate only)')
    args = parser.parse_args()

    delegate_opts = {}
    if args.ext_delegate_options:
        for item in args.ext_delegate_options.split(';'):
            kv = item.split(':')
            if len(kv) == 2:
                delegate_opts[kv[0].strip()] = kv[1].strip()

    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    interpreter, input_details, output_details = load_interpreter(
        args.model, args.ext_delegate, delegate_opts, args.num_threads)
    le_classes = np.load(args.labels, allow_pickle=True)
    print(f'Model ready. Classes: {list(le_classes)}')

    proc = psutil.Process()
    proc.cpu_percent(interval=None)   # prime the counter

    csv_path = 'features_data.csv'
    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)
    print(f'Writing dataset to {csv_path}')

    print(f'Opening {args.device}...')
    fd = os.open(args.device, os.O_RDWR)
    tty.setraw(fd)
    termios.tcflush(fd, termios.TCIFLUSH)

    os.write(fd, b'\x00')
    print(f'Handshake sent. '
          f'Header: {FRAME_HEADER_SIZE} B, features payload: {FEATURES_PAYLOAD_SIZE} B\n')
    print(f'{"seq":>6}  {"t_ms":>9}  {"label":<12}  {"pred":<12}  '
          f'{"t_infer":>8}  {"cpu%":>6}  '
          f'{"rms_x":>8}  {"rms_y":>8}  {"rms_z":>8}')
    print('-' * 100)

    prev_win_time = None

    while True:
        (magic, version, ftype, seq, ts_ms, label, flags, payload_len), payload_raw, skipped = \
            read_frame(fd)
        recv_ms = time.monotonic() * 1000.0

        if skipped:
            print(f'[warn] resync: skipped {skipped} byte(s)', file=sys.stderr)
        if version != FRAME_VERSION:
            print(f'[warn] unknown version {version}', file=sys.stderr)
            continue
        if ftype != FRAME_FEATURES:
            print(f'[warn] unexpected frame type {ftype}, skipping', file=sys.stderr)
            continue

        rms_x,  rms_y,  rms_z, \
        peak_x, peak_y, peak_z, \
        std_x,  std_y,  std_z, \
        p2p_x,  p2p_y,  p2p_z, \
        window_size, _reserved = struct.unpack(FEATURES_PAYLOAD_FORMAT, payload_raw)

        label_str = LABELS.get(label, f'UNKNOWN({label})')
        features  = [rms_x, rms_y, rms_z, peak_x, peak_y, peak_z,
                     std_x,  std_y,  std_z, p2p_x,  p2p_y,  p2p_z]

        prediction, t_infer_ms = predict(
            interpreter, input_details, output_details, le_classes, features)

        now = time.perf_counter()
        t_feat_ms          = 0.0   # M7 handled feature extraction
        t_window_ms        = t_infer_ms
        window_interval_ms = (now - prev_win_time) * 1000.0 if prev_win_time else 0.0
        prev_win_time      = now

        cpu_percent = proc.cpu_percent(interval=None)
        mem_rss_kb  = proc.memory_info().rss // 1024

        match = 'OK' if prediction == label_str else 'MISMATCH'

        writer.writerow([seq, ts_ms, f'{recv_ms:.3f}', label_str, prediction,
                         f'{rms_x:.3f}',  f'{rms_y:.3f}',  f'{rms_z:.3f}',
                         f'{peak_x:.3f}', f'{peak_y:.3f}', f'{peak_z:.3f}',
                         f'{std_x:.3f}',  f'{std_y:.3f}',  f'{std_z:.3f}',
                         f'{p2p_x:.3f}',  f'{p2p_y:.3f}',  f'{p2p_z:.3f}',
                         f'{t_feat_ms:.3f}', f'{t_infer_ms:.3f}', f'{t_window_ms:.3f}',
                         f'{window_interval_ms:.3f}', f'{cpu_percent:.1f}', mem_rss_kb])
        csv_file.flush()

        print(f'{seq:>6}  {ts_ms:>9}  {label_str:<12}  {prediction:<12}  '
              f'{t_infer_ms:>8.2f}  {cpu_percent:>6.1f}  '
              f'{rms_x:>8.1f}  {rms_y:>8.1f}  {rms_z:>8.1f}  [{match}]')


if __name__ == '__main__':
    main()
