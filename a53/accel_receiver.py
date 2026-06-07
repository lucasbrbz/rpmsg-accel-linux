#!/usr/bin/env python3
"""
Receives framed accelerometer data from the M7 core via RPMsg.
Each run writes a timestamped CSV: accel_data_YYYYMMDD_HHMMSS.csv

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
from datetime import datetime

RPMSG_DEVICE  = '/dev/ttyRPMSG0'

FRAME_MAGIC   = 0xA55AA55A
FRAME_VERSION = 1

# <I BB II BB H hhh
#  magic version type seq timestamp_ms label flags payload_len x y z
FRAME_FORMAT = '<IBBIIBBHhhh'
FRAME_SIZE   = struct.calcsize(FRAME_FORMAT)   # 24 bytes
MAGIC_LE     = struct.pack('<I', FRAME_MAGIC)  # b'\x5A\xA5\x5A\xA5'

CSV_HEADER = ['seq', 't_ms', 'recv_ms', 'label', 'x', 'y', 'z']

LABELS = {0: 'UNKNOWN', 1: 'NORMAL', 2: 'IMBALANCE', 3: 'ANOMALY'}
TYPES  = {1: 'RAW_ACCEL', 2: 'FEATURES', 3: 'STATUS'}

fd       = None
csv_file = None


def cleanup(signum, frame):
    print('\nInterrupted. Closing device.')
    if fd is not None:
        os.close(fd)
    if csv_file is not None:
        csv_file.close()
        print(f'Dataset saved.')
    sys.exit(0)


def read_exact(fd, n):
    buf = b''
    while len(buf) < n:
        chunk = os.read(fd, n - len(buf))
        if chunk:
            buf += chunk
    return buf


def read_frame(fd):
    """Scan byte-by-byte for the 4-byte MAGIC_LE, then bulk-read the rest.
    Returns (raw_bytes, skipped_count)."""
    skipped = 0
    window = b''
    while True:
        window += os.read(fd, 1)
        if len(window) < 4:
            continue
        if len(window) > 4:
            window = window[-4:]
        if window == MAGIC_LE:
            rest = read_exact(fd, FRAME_SIZE - 4)
            return MAGIC_LE + rest, skipped
        skipped += 1


def decode_flags(flags):
    parts = []
    if flags & 0x01:
        parts.append('mocked')
    if flags & 0x02:
        parts.append('calibrated')
    if flags & 0x04:
        parts.append('saturated')
    return ','.join(parts) if parts else 'none'


def main():
    global fd, csv_file

    parser = argparse.ArgumentParser(description='RPMsg accelerometer receiver')
    parser.add_argument('--device', default=RPMSG_DEVICE,
                        help=f'RPMsg tty device (default: {RPMSG_DEVICE})')
    csv_path = f'accel_data.csv'
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    csv_file = open(csv_path, 'w', newline='')
    writer   = csv.writer(csv_file)
    writer.writerow(CSV_HEADER)
    print(f'Writing dataset to {csv_path}')

    print(f'Opening {args.device}...')
    fd = os.open(args.device, os.O_RDWR)
    tty.setraw(fd)                         # disable all tty text processing (ICRNL, ONLCR, etc.)
    termios.tcflush(fd, termios.TCIFLUSH)  # discard stale buffered data

    # Send handshake so M7 captures our endpoint address and starts streaming.
    os.write(fd, b'\x00')
    print(f'Handshake sent. Frame size: {FRAME_SIZE} bytes. Waiting for data...\n')
    print(f'{"seq":>8}  {"t_ms":>9}  {"recv_ms":>14}  {"label":<12}  {"x":>7}  {"y":>7}  {"z":>7}  flags')
    print('-' * 84)

    while True:
        raw, skipped = read_frame(fd)
        recv_ms = time.monotonic() * 1000.0

        if skipped:
            print(f'[warn] resync: skipped {skipped} byte(s)', file=sys.stderr)

        _, version, ftype, seq, ts_ms, label, flags, plen, x, y, z = \
            struct.unpack(FRAME_FORMAT, raw)

        if version != FRAME_VERSION:
            print(f'[warn] unknown version {version}', file=sys.stderr)
            continue

        label_str = LABELS.get(label, f'UNKNOWN({label})')

        writer.writerow([seq, ts_ms, f'{recv_ms:.3f}', label_str, x, y, z])
        csv_file.flush()

        print(f'{seq:>8}  {ts_ms:>9}  {recv_ms:>14.3f}  {label_str:<12}  {x:>7}  {y:>7}  {z:>7}  {decode_flags(flags)}')


if __name__ == '__main__':
    main()
