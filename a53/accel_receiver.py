#!/usr/bin/env python3
"""
Receives simulated accelerometer data from the M7 core via RPMsg.

Usage:
    python3 accel_receiver.py [--device /dev/rpmsg0]
"""

import os
import signal
import struct
import sys
import argparse

RPMSG_DEVICE   = '/dev/ttyRPMSG0'
MSG_FORMAT     = '<hhhBB'   # x, y, z (int16), state (uint8), padding (uint8)
MSG_SIZE       = struct.calcsize(MSG_FORMAT)
STATES         = {0: 'NORMAL', 1: 'IMBALANCE', 2: 'ANOMALY'}

fd = None


def cleanup(signum, frame):
    print('\nInterrupted. Closing device.')
    if fd is not None:
        os.close(fd)
    sys.exit(0)


def main():
    global fd

    parser = argparse.ArgumentParser(description='RPMsg accelerometer receiver')
    parser.add_argument('--device', default=RPMSG_DEVICE,
                        help=f'RPMsg char device (default: {RPMSG_DEVICE})')
    args = parser.parse_args()

    signal.signal(signal.SIGINT,  cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    print(f'Opening {args.device}...')
    fd = os.open(args.device, os.O_RDWR)

    # Handshake: send one byte so the M7 captures our endpoint address
    # and starts streaming.
    os.write(fd, b'\x00')
    print('Handshake sent. Waiting for data...\n')
    print(f'{"State":<12} {"X":>8} {"Y":>8} {"Z":>8}')
    print('-' * 42)

    while True:
        raw = os.read(fd, MSG_SIZE)
        if len(raw) < MSG_SIZE:
            continue
        x, y, z, state, _ = struct.unpack(MSG_FORMAT, raw)
        label = STATES.get(state, f'UNKNOWN({state})')
        print(f'{label:<12} {x:>8}  {y:>8}  {z:>8}')


if __name__ == '__main__':
    main()
