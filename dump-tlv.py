#!/usr/bin/env python3

"""Pretty-print a tag-length-value (TLV) binary produced by TLVList.pack()."""

import struct
import sys

# Maximum number of lines to print in the hexdump
HEXDUMP_MAX_LINES = 4

# Number of bytes per line in the hexdump
BYTES_PER_LINE = 16

class MyException(Exception):
    pass

def hexdump(data: bytes, offset: int = 0) -> list[str]:
    lines = []
    for i in range(0, len(data), BYTES_PER_LINE):
        chunk = data[i:i + BYTES_PER_LINE]
        hex_parts = ' '.join(f'{b:02x}' for b in chunk)
        ascii_parts = ''.join(chr(b) if 0x20 <= b < 0x7f else '.' for b in chunk)
        lines.append(f'  {offset + i:08x}  {hex_parts:<{BYTES_PER_LINE * 3 - 1}}  |{ascii_parts}|')
    return lines

def dump_tlv(path: str):
    with open(path, 'rb') as f:
        blob = f.read()

    if len(blob) < 4:
        raise MyException(f'{path}: file too short for header')

    magic = struct.unpack_from('4s', blob, 0)[0]
    print(f'Header: "{magic.decode("ascii", errors="replace")}"')
    print()
    pos = 4

    index = 0
    while pos + 8 <= len(blob):
        tag_raw, length = struct.unpack_from('<4sI', blob, pos)
        tag = tag_raw.decode('ascii', errors='replace')
        pos += 8

        value = blob[pos:pos + length]
        pos += length
        pos += (-length) % 4  # skip padding

        print(f'[{index}] tag="{tag}" length={length}')
        lines = hexdump(value)
        if len(lines) <= HEXDUMP_MAX_LINES:
            print('\n'.join(lines))
        else:
            print('\n'.join(lines[:HEXDUMP_MAX_LINES]))
            remaining = len(lines) - HEXDUMP_MAX_LINES
            print(f'  ... ({remaining} more line{"s" if remaining != 1 else ""})')
        print()
        index += 1

if __name__ == '__main__':
    try:
        if len(sys.argv) != 2:
            raise MyException(f'Usage: {sys.argv[0]} <tlv-file>')
        dump_tlv(sys.argv[1])
    except MyException as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)
