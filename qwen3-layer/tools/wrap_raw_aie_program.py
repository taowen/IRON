#!/usr/bin/env python3
"""Wrap raw AIE2P program bytes as a loadable ELF32 executable."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


ELF32_EHDR_SIZE = 52
ELF32_PHDR_SIZE = 32
ELF32_SHDR_SIZE = 40
EM_AIE = 0x108
EF_AIE_AIE2P = 0x3


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def _phdr(
    p_type: int,
    offset: int,
    vaddr: int,
    paddr: int,
    filesz: int,
    memsz: int,
    flags: int,
    align: int,
) -> bytes:
    return struct.pack("<IIIIIIII", p_type, offset, vaddr, paddr, filesz, memsz, flags, align)


def _shdr(
    name: int,
    sh_type: int,
    flags: int,
    addr: int,
    offset: int,
    size: int,
    link: int,
    info: int,
    addralign: int,
    entsize: int,
) -> bytes:
    return struct.pack(
        "<IIIIIIIIII",
        name,
        sh_type,
        flags,
        addr,
        offset,
        size,
        link,
        info,
        addralign,
        entsize,
    )


def wrap_raw_program(raw: bytes, text_addr: int = 0) -> bytes:
    shstr = b"\x00.text\x00.shstrtab\x00"
    text_name = shstr.index(b".text")
    shstr_name = shstr.index(b".shstrtab")

    phoff = ELF32_EHDR_SIZE
    text_off = _align_up(ELF32_EHDR_SIZE + ELF32_PHDR_SIZE, 16)
    shstr_off = text_off + len(raw)
    shoff = _align_up(shstr_off + len(shstr), 4)

    out = bytearray()
    out.extend(b"\x7fELF")
    out.extend(bytes([1, 1, 1, 0]))
    out.extend(b"\x00" * 8)
    out.extend(
        struct.pack(
            "<HHIIIIIHHHHHH",
            2,
            EM_AIE,
            1,
            text_addr,
            phoff,
            shoff,
            EF_AIE_AIE2P,
            ELF32_EHDR_SIZE,
            ELF32_PHDR_SIZE,
            1,
            ELF32_SHDR_SIZE,
            3,
            2,
        )
    )
    out.extend(_phdr(1, text_off, text_addr, text_addr, len(raw), len(raw), 0x5, 16))
    out.extend(b"\x00" * (text_off - len(out)))
    out.extend(raw)
    out.extend(shstr)
    out.extend(b"\x00" * (shoff - len(out)))
    out.extend(_shdr(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    out.extend(_shdr(text_name, 1, 0x6, text_addr, text_off, len(raw), 0, 0, 16, 0))
    out.extend(_shdr(shstr_name, 3, 0, 0, shstr_off, len(shstr), 0, 0, 1, 0))
    return bytes(out)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--text-addr", type=lambda value: int(value, 0), default=0)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    args.output.write_bytes(wrap_raw_program(args.input.read_bytes(), args.text_addr))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
