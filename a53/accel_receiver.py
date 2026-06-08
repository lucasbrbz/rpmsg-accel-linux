#!/usr/bin/env python3
"""
Receives FRAME_FEATURES frames from the M7 core via RPMsg and writes features_data.csv.

The M7 accumulates FEATURE_WINDOW_SIZE raw samples, computes RMS / peak / std dev /
peak-to-peak per axis, and sends one features frame per window (2 s at 100 ms/sample).

Usage:
    python3 accel_receiver.py [--device /dev/ttyRPMSG0]
"""

import csv
import os
import signal
import struct
import sys
import argparse
import termios
import time
import tty

RPMSG_DEVICE  = '/dev/ttyRPMSG0'
FRAME_MAGIC   = 0xA55AA55A
FRAME_VERSION = 1

MAGIC_LE = struct.pack('<I', FRAME_MAGIC)

# Header layout: magic(I) version(B) type(B) seq(I) ts_ms(I) label(B) flags(B) payload_len(H)
FRAME_HEADER_FORMAT = '<IBBIIBBH'
FRAME_HEADER_SIZE   = struct.calcsize(FRAME_HEADER_FORMAT)   # 18 bytes

# Features payload: rms_x/y/z, peak_x/y/z, std_x/y/z, p2p_x/y/z (12 floats), window_size(H), reserved(H)
FEATURES_PAYLOAD_FORMAT = '<' + 'f' * 12 + 'HH'
FEATURES_PAYLOAD_SIZE   = struct.calcsize(FEATURES_PAYLOAD_FORMAT)  # 52 bytes

FRAME_RAW_ACCEL = 1
FRAME_FEATURES  = 2
FRAME_STATUS    = 3

CSV_HEADER = ['seq', 't_ms', 'recv_ms', 'label',
              'rms_x',  'rms_y',  'rms_z',
              'peak_x', 'peak_y', 'peak_z',
              'std_x',  'std_y',  'std_z',
              'p2p_x',  'p2p_y',  'p2p_z']

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
    """Scan for the 4-byte magic, then read the header and variable-length payload."""
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
    payload_len = fields[7]          # index of payload_len in the unpacked tuple
    payload_raw = read_exact(fd, payload_len)
    return fields, payload_raw, skipped


def main():
    global fd, csv_file

    parser = argparse.ArgumentParser(description='RPMsg feature frame receiver')
    parser.add_argument('--device', default=RPMSG_DEVICE,
                        help=f'RPMsg tty device (default: {RPMSG_DEVICE})')
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

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
    print(f'{"seq":>6}  {"t_ms":>9}  {"recv_ms":>14}  {"label":<12}  '
          f'{"rms_x":>8}  {"rms_y":>8}  {"rms_z":>8}  '
          f'{"p2p_x":>8}  {"p2p_y":>8}  {"p2p_z":>8}')
    print('-' * 106)

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

        writer.writerow([seq, ts_ms, f'{recv_ms:.3f}', label_str,
                         f'{rms_x:.3f}',  f'{rms_y:.3f}',  f'{rms_z:.3f}',
                         f'{peak_x:.3f}', f'{peak_y:.3f}', f'{peak_z:.3f}',
                         f'{std_x:.3f}',  f'{std_y:.3f}',  f'{std_z:.3f}',
                         f'{p2p_x:.3f}',  f'{p2p_y:.3f}',  f'{p2p_z:.3f}'])
        csv_file.flush()

        print(f'{seq:>6}  {ts_ms:>9}  {recv_ms:>14.3f}  {label_str:<12}  '
              f'{rms_x:>8.1f}  {rms_y:>8.1f}  {rms_z:>8.1f}  '
              f'{p2p_x:>8.1f}  {p2p_y:>8.1f}  {p2p_z:>8.1f}')


if __name__ == '__main__':
    main()
