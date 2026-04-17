#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Converts OpenRM binhex-encoded images to Nova-compatible binary blobs.

import sys
import os
import argparse
import re
import gzip
import struct
import tempfile
import urllib.request

FLCN_BLK_ALIGNMENT = 256

class MyException(Exception):
    pass

# -------------------------------------------------------------------
# Parse binhex arrays from OpenRM
# -------------------------------------------------------------------

def parse_array(f):
    """Parses a bindata array definition and returns its binhex as bytes

    Example:
    static BINDATA_CONST NvU8 ksec2BinArchiveSecurescrubUcode_AD10X_header_prod_data[] =
    {
        0x63, 0x60, 0x00, 0x02, 0x46, 0x20, 0x96, 0x02, 0x62, 0x66, 0x08, 0x13, 0x4c, 0x48, 0x42, 0x69,
        0x20, 0x00, 0x00, 0x30, 0x39, 0x0a, 0xfc, 0x24, 0x00, 0x00, 0x00,
    };
    """
    output = b''
    for line in f:
        if "};" in line:
            break
        bytes = [int(b, 16) for b in re.findall('0x[0-9a-f][0-9a-f]', line)]
        if len(bytes) > 0:
            output += struct.pack(f"{len(bytes)}B", *bytes)

    return output

def parse_struct(f):
    """Parses a struct definition and returns its binhex as bytes

    Example:
    static const RM_FLCN_BL_DESC ksec2BinArchiveBlUcode_TU102_ucode_desc_data = {
        0xfd,
        0,
        {
            0x0,
            0x200,
            0x200,
            0x100
        }
    };

    """
    output = b''
    for line in f:
        if "};" in line:
            break
        words = [int(b, 16) for b in re.findall('(?:0x|)[0-9a-f]+', line)]
        if len(words) > 0:
            output += struct.pack(f"<{len(words)}I", *words)


    return output

def get_bytes(filename, array1, array2):
    """Extract the bytes for the given array or struct in the given file.

    :param filename: the file to parse
    :param array1: the first half of name of the array/struct to parse
    :param array2: the second half
    :returns: byte array

    This function scans the file for the array or struct and returns a bytearray
    of its contents, uncompressing the data if it is tagged as compressed.

    This function assumes that each array/struct is immediately preceded with a
    comment section that specifies whether the array is compressed and how many
    bytes of data there should be.  Example:

    //
    // FUNCTION: ksec2GetBinArchiveSecurescrubUcode_AD10X("header_prod")
    // FILE NAME: kernel/inc/securescrub/bin/ad10x/g_securescrubuc_sec2_ad10x_boot_from_hs_prod.h
    // FILE TYPE: TEXT
    // VAR NAME: securescrub_ucode_header_ad10x_boot_from_hs
    // COMPRESSION: YES
    // COMPLEX_STRUCT: NO
    // DATA SIZE (bytes): 36
    // COMPRESSED SIZE (bytes): 27
    //
    static BINDATA_CONST NvU8 ksec2BinArchiveSecurescrubUcode_AD10X_header_prod_data[] =

    The actual extraction of binhex bytes is handled by parse_array() or parse_struct().
    """

    # Build the five possible array/struct names.  BINDATA_LABEL was added in r575,
    # and NV_DECLARE_ALIGNED(NvU8, 8) was added in r590.
    arrays = [
        f"static BINDATA_CONST NvU8 {array1}_{array2}_data",
        f"static BINDATA_CONST NvU8 {array1}_BINDATA_LABEL_{array2.upper()}_data",
        f"static BINDATA_CONST NV_DECLARE_ALIGNED(NvU8, 8) {array1}_BINDATA_LABEL_{array2.upper()}_data",
        f"static const {array1}_{array2}_data",
        f"static const {array1}_BINDATA_LABEL_{array2.upper()}_data",
    ]

    with open(filename) as f:
        for line in f:
            m = re.search(r"COMPRESSION: (\w*)", line)
            if m:
                is_compressed = m.group(1) == "YES"
            m = re.search(r"COMPLEX_STRUCT: (\w*)", line)
            if m:
                is_struct = m.group(1) == "YES"
            m = re.search(r"DATA SIZE \(bytes\): (\d+)", line)
            if m:
                data_size = int(m.group(1))
            m = re.search(r"DATA SIZE \(bytes\): sizeof\((\d+)\)", line)
            if m:
                data_size = None
            m = re.search(r"COMPRESSED SIZE \(bytes\): N/A", line)
            if m:
                compressed_size = None
            m = re.search(r"COMPRESSED SIZE \(bytes\): (\d+)", line)
            if m:
                compressed_size = int(m.group(1))
            m = next((a for a in arrays if a in line), None)
            if m:
                # We found the array, so remember its name in case we need to report an error
                array = m
                break
        else:
            raise MyException(f"array {array1}_{array2}_data not found in {filename}")

        if is_struct:
            output = parse_struct(f)
            # Struct entries reference themselves for the size.  The only way
            # to determine the actual size is to compile the C code.  Instead,
            # just assume the header file is complete.
            data_size = len(output)
        else:
            output = parse_array(f)

    if len(output) == 0:
        raise MyException(f"no data found for {array} in {filename}")

    # Structs are never compressed
    if is_struct and is_compressed:
        raise MyException(f"struct {array} in {filename} cannot be compressed")

    # Make sure we actually read a compressed size
    if is_compressed and not compressed_size:
        raise MyException(f"array {array} in {filename} compressed size is undetermined")

    if is_compressed:
        if len(output) != compressed_size:
            raise MyException(f"compressed array {array} in {filename} should be {compressed_size} bytes but is actually {len(output)}.")
        gzipheader = struct.pack("<4BL2B", 0x1f, 0x8b, 8, 0, 0, 0, 3)
        output = gzip.decompress(gzipheader + output)
        if len(output) != data_size:
            raise MyException(f"array {array} in {filename} decompressed to {len(output)} bytes but should have been {data_size} bytes.")
        return output
    else:
        if len(output) != data_size:
            raise MyException(f"array {array} in {filename} should be {data_size} bytes but is actually {len(output)}.")
        return output

# -------------------------------------------------------------------
# Build tag-length-value (TLV) list
# -------------------------------------------------------------------

class TLV:
    def __init__(self, filename: str, gpu: str):
        global version
        self.filename = filename
        self.gpu = gpu
        self.entries = [("VERS", version)]

    def add(self, tag: str, value):
        if len(tag) != 4:
            raise MyException(f"TLV tag '{tag}' must be exactly 4 characters")

        # Integers are a special case, as they have no "length" in Python
        if isinstance(value, int):
            # For simplicity, we only support 32-bit unsigned integers
            if not (0 <= value <= 0xFFFFFFFF):
                raise MyException(f"TLV tag '{tag}' integer value {value} out of uint32 range")
        else:
            if len(value) == 0:
                raise MyException(f"TLV tag '{tag}' as no data")
            # We don't want non-ASCII strings anywhere
            if isinstance(value, str) and not value.isascii():
                raise MyException(f"TLV tag '{tag}' value is a string but contains non-ASCII characters")

        self.entries.append((tag, value))

    def write(self):
        global outputpath

        print(f"Creating nvidia/{self.gpu}/gsp/{self.filename}.tlv")
        os.makedirs(f"{outputpath}/nvidia/{self.gpu}/gsp/", exist_ok = True)

        with open(f"{outputpath}/nvidia/{self.gpu}/gsp/{self.filename}.tlv", "wb") as f:
            f.write(struct.pack('4s', b"FWPM"))

            for tag, value in self.entries:
                # Convert strings and integers into bytearrays
                if isinstance(value, str):
                    value = value.encode('ascii')
                elif isinstance(value, int):
                    value = struct.pack('<I', value)

                f.write(struct.pack('<4sI', tag.encode('ascii'), len(value)))
                f.write(value)
                # Add padding bytes if necessary
                f.write(b'\x00' * ((-len(value)) % 4))


# -------------------------------------------------------------------
# Read ELF images
# -------------------------------------------------------------------

class ELF64:
    EI_NIDENT = 16
    ELF_MAGIC = b'\x7fELF'
    ELFCLASS64 = 2

    def __init__(self, filename: str):
        with open(filename, 'rb') as f:
            data = f.read()

        if len(data) < self.EI_NIDENT:
            raise MyException(f"{filename}: file too small to be an ELF image")

        if data[:4] != self.ELF_MAGIC:
            raise MyException(f"{filename}: not an ELF file")

        if data[4] != self.ELFCLASS64:
            raise MyException(f"{filename}: not a 64-bit ELF file")

        if data[5] != 1:
            raise MyException(f"{filename}: big-endian ELF files are not supported")

        # self.filename = filename

        (e_shoff, e_shentsize, e_shnum, e_shstrndx) = struct.unpack_from('<Q10xHHH', data, 0x28)

        shstrtab_off = struct.unpack_from('<24xQ', data,
            e_shoff + e_shstrndx * e_shentsize)[0]

        self.sections = []
        for i in range(e_shnum):
            sh = e_shoff + i * e_shentsize
            (sh_name, sh_offset, sh_size) = struct.unpack_from('<I20xQQ', data, sh)

            name_start = shstrtab_off + sh_name
            name_end = data.index(b'\x00', name_start)
            name = data[name_start:name_end].decode('ascii')

            # We only care about sections that start with ".fw"
            if name.startswith('.fw'):
                self.sections.append((name, data[sh_offset:sh_offset + sh_size]))

    def section(self, name: str) -> bytes:
        for sec_name, sec_data in self.sections:
            if sec_name == name:
                return sec_data
        raise MyException(f"ELF {self.filename} does not have a {name} section")

# -------------------------------------------------------------------
# Generate firmware binaries
# -------------------------------------------------------------------

def round_up_to_base(x, base = 10):
    return x + (base - x) % base

# Generic Falcon bootloader.  First, FWSEC runs on the RISC-V GSP core.
# Then this generic bootloader runs on the SEC2 core, in order to restart the GSP
# core to run GSP-RM on it.  This is only used on TU10x and GA100 GPUs.
def generic_bootloader(gpu):
    global outputpath
    global version

    GPU = gpu.upper()
    filename = f"src/nvidia/generated/g_bindata_ksec2GetBinArchiveBlUcode_{GPU}.c"

    tlv = TLV("gen_bootloader", gpu)

    # Extract the descriptor (RM_FLCN_BL_DESC)
    descriptor = get_bytes(filename, f"RM_FLCN_BL_DESC ksec2BinArchiveBlUcode_{GPU}", "ucode_desc")
    tlv.add("DESC", descriptor)

    # Extract the actual bootloader firmware
    firmware = get_bytes(filename, f"ksec2BinArchiveBlUcode_{GPU}", "ucode_image")
    tlv.add("BLOB", firmware)

    tlv.write()

# GSP bootloader
def gsp_bootloader(gpu: str, fuse = ""):
    global outputpath
    global version

    # Prepend an underscore if not empty
    if len(fuse) > 0:
        fuse = f"_{fuse}"

    GPU = gpu.upper()
    filename = f"src/nvidia/generated/g_bindata_kgspGetBinArchiveGspRmBoot_{GPU}.c"

    tlv = TLV("gsp_bootloader", gpu)

    # Extract the descriptor (RM_RISCV_UCODE_DESC)
    # Note: the size of RM_RISCV_UCODE_DESC varies from version to version, but Nova
    # only cares about the first few fields.
    descriptor = get_bytes(filename, f"kgspBinArchiveGspRmBoot_{GPU}", f"ucode_desc{fuse}")
    tlv.add("DESC", descriptor)

    # Extract the actual bootloader firmware
    firmware = get_bytes(filename, f"kgspBinArchiveGspRmBoot_{GPU}", f"ucode_image{fuse}")
    tlv.add("BLOB", firmware)

    tlv.write()

# GSP Booter load and unload
def booter(gpu, load, sigsize, fuse = "prod"):
    global outputpath
    global version

    GPU = gpu.upper()
    LOAD = load.capitalize()
    name = f"booter-{load}-{gpu}-{fuse}"

    filename = f"src/nvidia/generated/g_bindata_kgspGetBinArchiveBooter{LOAD}Ucode_{GPU}.c"

    tlv = TLV("booter", gpu)

    # Query the number of signatures.  This should be a 4-byte array (32-bit little-endian integer)
    bytes = get_bytes(filename, f"kgspBinArchiveBooter{LOAD}Ucode_{GPU}", "num_sigs")
    if len(bytes) != 4:
        raise MyException(f"num_sigs array for {name} is wrong size of {len(bytes)}")
    num_sigs = struct.unpack("<I", bytes)[0]
    if num_sigs < 1 or num_sigs > 15:
        raise MyException(f"out of range number of signatures ({num_sigs}) for {name}")
    tlv.add("NSIG", num_sigs)

    # Extract the signatures.  Technically, we don't need to pass the signature size to
    # this function, but doing so allows us to double-check all the array sizes.
    signatures = get_bytes(filename, f"kgspBinArchiveBooter{LOAD}Ucode_{GPU}", f"sig_{fuse}")
    signatures_size = len(signatures)
    if signatures_size % sigsize:
        raise MyException(f"signature file size for {name} is {signatures_size}, an uneven multiple of {sigsize}")
    if num_sigs != signatures_size // sigsize:
        raise MyException(f"mismatch number of signatures ({signatures_size // sigsize}), should be {num_sigs}")
    tlv.add("SIGS", signatures)

    # Extract the patch location
    bytes = get_bytes(filename, f"kgspBinArchiveBooter{LOAD}Ucode_{GPU}", "patch_loc")
    if len(bytes) != 4:
        raise MyException(f"patch_loc[] array for {name} should be one one element, but is {len(bytes)} bytes.")
    patchloc = struct.unpack("<I", bytes)[0]
    tlv.add("PLOC", patchloc)

    # Extract the patch sig offset.  RM expects this to be zero, but doesn't use it,
    # so if it's ever non-zero, something has changed.
    bytes = get_bytes(filename, f"kgspBinArchiveBooter{LOAD}Ucode_{GPU}", "patch_sig")
    if len(bytes) != 4:
        raise MyException(f"patch_sig[] array for {name} should be one one element, but is {len(bytes)} bytes.")
    patchsig = struct.unpack("<I", bytes)[0]
    if patchsig != 0:
        raise MyException(f"patch_sig for {name} should be 0, but is instead {patchsig}.")

    # Extract the patch meta variables
    bytes = get_bytes(filename, f"kgspBinArchiveBooter{LOAD}Ucode_{GPU}", "patch_meta")
    fuse_ver, engine_id, ucode_id = struct.unpack("<III", bytes)
    tlv.add("FUSE", fuse_ver)
    tlv.add("ENID", engine_id)
    tlv.add("UCID", ucode_id)

    # Extract the descriptor (nvfw_hs_load_header_v2)
    descriptor = get_bytes(filename, f"kgspBinArchiveBooter{LOAD}Ucode_{GPU}", f"header_{fuse}")

    # Extract some of individual fields of nvfw_hs_load_header_v2
    # num_apps is the fifth field of struct nvfw_hs_load_header_v2
    (os_code_offset, os_code_size, os_data_offset, os_data_size, num_apps,
     app_code_offset, app_code_size, app_data_offset, app_data_size) = struct.unpack("<9I", descriptor)
    # Verify that sizeof(descriptor) == 5 * 4 + num_apps * 16
    if len(descriptor) != 5 * 4 + num_apps * 16:
        raise MyException(f"nvfw_hs_load_header_v2 descriptor for {name} should be {5 * 4 + num_apps * 16} bytes, but is instead {len(descriptor)} bytes.")
    # Nova depends on os_code_size == app_code_offset
    if os_code_size != app_code_offset:
        raise MyException(f"nvfw_hs_load_header_v2 descriptor for {name} has os_code_size={os_code_size} and app_code_offset={app_code_offset}, but they should be the same.")

    tlv.add("CDOF", os_code_offset)
    tlv.add("CDSZ", os_code_size)
    tlv.add("DAOF", os_data_offset)
    tlv.add("DASZ", os_data_size)
    tlv.add("APOF", app_code_offset)
    tlv.add("APSZ", app_code_size)

    # Extract the actual booter firmware
    firmware = get_bytes(filename, f"kgspBinArchiveBooter{LOAD}Ucode_{GPU}", f"image_{fuse}")
    tlv.add("BLOB", firmware)

    tlv.write()

# GPU memory scrubber, needed for some GPUs and configurations
def scrubber(gpu, sigsize, fuse = "prod"):
    global outputpath
    global version

    # Unfortunately, RM breaks convention with the scrubber image and labels
    # the files and arrays with AD10X instead of AD102.
    GPUX = f"{gpu[:-1].upper()}X"
    name = f"scrubber-{gpu}-{fuse}"

    filename = f"src/nvidia/generated/g_bindata_ksec2GetBinArchiveSecurescrubUcode_{GPUX}.c"

    tlv = TLV("scrubber", gpu)

    # Query the number of signatures.  This should be a 4-byte array (32-bit little-endian integer)
    bytes = get_bytes(filename, f"ksec2BinArchiveSecurescrubUcode_{GPUX}", "num_sigs")
    if len(bytes) != 4:
        raise MyException(f"num_sigs array for {name} is wrong size of {len(bytes)}")
    num_sigs = struct.unpack("<I", bytes)[0]
    if num_sigs < 1 or num_sigs > 15:
        raise MyException(f"out of range number of signatures ({num_sigs}) for {name}")
    tlv.add("NSIG", num_sigs)

    # Extract the signatures.  Technically, we don't need to pass the signature size to
    # this function, but doing so allows us to double-check all the array sizes.
    signatures = get_bytes(filename, f"ksec2BinArchiveSecurescrubUcode_{GPUX}", f"sig_{fuse}")
    signatures_size = len(signatures)
    if signatures_size % sigsize:
        raise MyException(f"signature file size for {name} is {signatures_size}, an uneven multiple of {sigsize}")
    if num_sigs != signatures_size // sigsize:
        raise MyException(f"mismatch number of signatures ({signatures_size // sigsize}), should be {num_sigs}")
    tlv.add("SIGS", signatures)

    # Extract the patch location
    bytes = get_bytes(filename, f"ksec2BinArchiveSecurescrubUcode_{GPUX}", "patch_loc")
    if len(bytes) != 4:
        raise MyException(f"patch_loc[] array for {name} should be one one element, but is {len(bytes)} bytes.")
    patchloc = struct.unpack("<I", bytes)[0]
    tlv.add("PLOC", patchloc)

    # Extract the patch sig offset.  RM expects this to be zero, but doesn't use it,
    # so if it's ever non-zero, something has changed.
    bytes = get_bytes(filename, f"ksec2BinArchiveSecurescrubUcode_{GPUX}", "patch_sig")
    if len(bytes) != 4:
        raise MyException(f"patch_sig[] array for {name} should be one one element, but is {len(bytes)} bytes.")
    patchsig = struct.unpack("<I", bytes)[0]
    if patchsig != 0:
        raise MyException(f"patch_sig for {name} should be 0, but is instead {patchsig}.")

    # Extract the patch meta variables
    bytes = get_bytes(filename, f"ksec2BinArchiveSecurescrubUcode_{GPUX}", "patch_meta")
    fuse_ver, engine_id, ucode_id = struct.unpack("<III", bytes)
    tlv.add("FUSE", fuse_ver)
    tlv.add("ENID", engine_id)
    tlv.add("UCID", ucode_id)

    # Extract the descriptor (nvfw_hs_load_header_v2)
    descriptor = get_bytes(filename, f"ksec2BinArchiveSecurescrubUcode_{GPUX}", f"header_{fuse}")

    # Extract some of individual fields of nvfw_hs_load_header_v2
    # num_apps is the fifth field of struct nvfw_hs_load_header_v2
    (os_code_offset, os_code_size, os_data_offset, os_data_size, num_apps,
     app_code_offset, app_code_size, app_data_offset, app_data_size) = struct.unpack("<9I", descriptor)
    # Verify that sizeof(descriptor) == 5 * 4 + num_apps * 16
    if len(descriptor) != 5 * 4 + num_apps * 16:
        raise MyException(f"nvfw_hs_load_header_v2 descriptor for {name} should be {5 * 4 + num_apps * 16} bytes, but is instead {len(descriptor)} bytes.")
    # Nova depends on os_code_size == app_code_offset
    if os_code_size != app_code_offset:
        raise MyException(f"nvfw_hs_load_header_v2 descriptor for {name} has os_code_size={os_code_size} and app_code_offset={app_code_offset}, but they should be the same.")

    tlv.add("CDOF", os_code_offset)
    tlv.add("CDSZ", os_code_size)
    tlv.add("DAOF", os_data_offset)
    tlv.add("DASZ", os_data_size)
    tlv.add("APOF", app_code_offset)
    tlv.add("APSZ", app_code_size)

    # Extract the actual scrubber firmware
    firmware = get_bytes(filename, f"ksec2BinArchiveSecurescrubUcode_{GPUX}", f"image_{fuse}")
    tlv.add("BLOB", firmware)

    tlv.write()

# Unlike the other images, FMC firmware and its metadata are encapsulated in
# an ELF image.  FMC metadata is simpler than the other firmware types, as it
# comprises just three binary blobs.
def fmc(gpu: str, fuse: str):
    global outputpath
    global version

    GPU=gpu.upper()
    filename = f"src/nvidia/generated/g_bindata_kgspGetBinArchiveGspRmFmcGfw{fuse}Signed_{GPU}.c"

    tlv = TLV("fmc", gpu)

    ucode_hash = get_bytes(filename, f"kgspBinArchiveGspRmFmcGfw{fuse}Signed_{GPU}", "ucode_hash")
    tlv.add("HASH", ucode_hash)

    ucode_sig = get_bytes(filename, f"kgspBinArchiveGspRmFmcGfw{fuse}Signed_{GPU}", "ucode_sig")
    tlv.add("SIGS", ucode_sig)

    ucode_pkey = get_bytes(filename, f"kgspBinArchiveGspRmFmcGfw{fuse}Signed_{GPU}", "ucode_pkey")
    tlv.add("PKEY", ucode_pkey)

    ucode_image = get_bytes(filename, f"kgspBinArchiveGspRmFmcGfw{fuse}Signed_{GPU}", "ucode_image")
    tlv.add("BLOB", ucode_image)

    tlv.write()

def fwimage_from_gsp_elf(filename: str, gpu: str):
    global outputpath

    elf = ELF64(filename)

    with open(f"{outputpath}/nvidia/{gpu}/gsp/gsp.bin", "wb") as f:
        f.write(elf.section(".fwimage"))

    print(f"Created {gpu}/gsp/gsp.bin from {filename}")

# Generate a gsp.tlv file that points to the correct GSP image
# `elf` is the original ELF image from the .run file or build
# `signame` is the name of the .fwsignature section to extract
# `gpu` is the GPU name
def gsp_tlv_from_elf(elf: ELF64, signame: str, gpu: str):
    signature = elf.section(signame)

    tlv = TLV("gsp", gpu)
    tlv.add("SIGN", signature)
    tlv.add("SIZE", len(elf.section(".fwimage")))
    tlv.add("FILE", "gsp.bin")
    tlv.write()

# Extract the GSP-RM firmware from the .run file and copy the binaries
# to the target directory.
def gsp_firmware_from_run(filename):
    global outputpath
    global version

    import subprocess
    import shutil

    basename = os.path.basename(filename)

    with tempfile.TemporaryDirectory() as temp:
        os.chdir(temp)

        try:
            print(f"Validating {basename}")

            result = subprocess.run(['/bin/sh', filename, '--check'], shell=False,
                                    check=True, timeout=10,
                                    stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
            output = result.stdout.strip().decode("ascii")
            if not "check sums and md5 sums are ok" in output:
                raise MyException(f"{basename} is not a valid Nvidia driver .run file")
        except subprocess.CalledProcessError as error:
            print(error.output.decode())
            raise

        try:
            print(f"Extracting {basename} to {temp}")
            # The -x parameter tells the installer to only extract the
            # contents and then exit.
            subprocess.run(['/bin/sh', filename, '-x'], shell=False,
                           check=True, timeout=60,
                           stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
        except subprocess.SubprocessError as error:
            print(error.output.decode())
            raise

        try:
            # The .run file extracts its contents to a directory with the same
            # name as the file itself, minus the .run.  The GSP-RM firmware
            # images are in the 'firmware' subdirectory.
            result = subprocess.run(['/bin/sh', filename, '--target-directory'], shell=False,
                                    check=True, timeout=10,
                                    stdout = subprocess.PIPE, stderr = subprocess.DEVNULL)
            directory = result.stdout.strip().decode("ascii")
            os.chdir(f"{directory}/firmware")
        except subprocess.SubprocessError as e:
            print(e.output.decode())
            raise

        if not os.path.exists('gsp_tu10x.bin') or not os.path.exists('gsp_ga10x.bin'):
            raise MyException(f"Firmware files are missing in {basename}")

        fwimage_from_gsp_elf("gsp_tu10x.bin", "tu102")
        fwimage_from_gsp_elf("gsp_ga10x.bin", "ga102")

        elf = ELF64("gsp_tu10x.bin")
        gsp_tlv_from_elf(elf, ".fwsignature_tu10x", "tu102")
        gsp_tlv_from_elf(elf, ".fwsignature_tu11x", "tu116")
        gsp_tlv_from_elf(elf, ".fwsignature_ga100", "ga100")

        elf = ELF64("gsp_ga10x.bin")
        gsp_tlv_from_elf(elf, ".fwsignature_ga10x", "ga102")
        gsp_tlv_from_elf(elf, ".fwsignature_ad10x", "ad102")
        gsp_tlv_from_elf(elf, ".fwsignature_gh100", "gh100")
        gsp_tlv_from_elf(elf, ".fwsignature_gb10x", "gb100")
        gsp_tlv_from_elf(elf, ".fwsignature_gb20x", "gb202")

        # Copy ucodes binaries if present (r610+).  Each ucodes.bin is paired
        # with the corresponding gsp.bin and loaded separately by the driver.
        if os.path.exists('ucodes_tu10x.bin'):
            shutil.copyfile('ucodes_tu10x.bin', f"{outputpath}/nvidia/tu102/gsp/ucodes-{version}.bin")
            print(f"Copied ucodes_tu10x.bin to tu102/gsp/ucodes-{version}.bin")
        if os.path.exists('ucodes_ga10x.bin'):
            shutil.copyfile('ucodes_ga10x.bin', f"{outputpath}/nvidia/ga102/gsp/ucodes-{version}.bin")
            print(f"Copied ucodes_ga10x.bin to ga102/gsp/ucodes-{version}.bin")

# Extract GSP firmware from a local build output directory.
# This is an NVIDIA-internal feature for use with internal build systems.
def gsp_firmware_from_build(gsp_build_dir):
    global outputpath
    global version

    import shutil

    if not os.path.isdir(gsp_build_dir):
        raise MyException(f"GSP build directory does not exist: {gsp_build_dir}")

    tu10x_src = os.path.join(gsp_build_dir, "gsp_tu10x.bin")
    ga10x_src = os.path.join(gsp_build_dir, "gsp_ga10x.bin")

    if not os.path.exists(tu10x_src):
        raise MyException(f"GSP firmware not found: {tu10x_src}")
    if not os.path.exists(ga10x_src):
        raise MyException(f"GSP firmware not found: {ga10x_src}")

    fwimage_from_gsp_elf("gsp_tu10x.bin", "tu102")
    fwimage_from_gsp_elf("gsp_ga10x.bin", "ga102")

    elf = ELF64(tu10x_src)
    gsp_tlv_from_elf(elf, ".fwsignature_tu10x", "tu102")
    gsp_tlv_from_elf(elf, ".fwsignature_tu11x", "tu116")
    gsp_tlv_from_elf(elf, ".fwsignature_ga100", "ga100")

    elf = ELF64(ga10x_src)
    gsp_tlv_from_elf(elf, ".fwsignature_ga10x", "ga102")
    gsp_tlv_from_elf(elf, ".fwsignature_ad10x", "ad102")
    gsp_tlv_from_elf(elf, ".fwsignature_gh100", "gh100")
    gsp_tlv_from_elf(elf, ".fwsignature_gb10x", "gb100")
    gsp_tlv_from_elf(elf, ".fwsignature_gb20x", "gb202")


    os.makedirs(f"{outputpath}/nvidia/tu102/gsp/", exist_ok = True)
    os.makedirs(f"{outputpath}/nvidia/ga102/gsp/", exist_ok = True)

    shutil.copyfile(tu10x_src, f"{outputpath}/nvidia/tu102/gsp/gsp-{version}.bin")
    print(f"Copied gsp_tu10x.bin to nvidia/tu102/gsp/gsp-{version}.bin")

    shutil.copyfile(ga10x_src, f"{outputpath}/nvidia/ga102/gsp/gsp-{version}.bin")
    print(f"Copied gsp_ga10x.bin to nvidia/ga102/gsp/gsp-{version}.bin")

    # Copy ucodes binaries if present (r610+)
    ucodes_tu10x_src = os.path.join(gsp_build_dir, "ucodes_tu10x.bin")
    ucodes_ga10x_src = os.path.join(gsp_build_dir, "ucodes_ga10x.bin")

    if os.path.exists(ucodes_tu10x_src):
        shutil.copyfile(ucodes_tu10x_src, f"{outputpath}/nvidia/tu102/gsp/ucodes-{version}.bin")
        print(f"Copied ucodes_tu10x.bin to nvidia/tu102/gsp/ucodes-{version}.bin")

    if os.path.exists(ucodes_ga10x_src):
        shutil.copyfile(ucodes_ga10x_src, f"{outputpath}/nvidia/ga102/gsp/ucodes-{version}.bin")
        print(f"Copied ucodes_ga10x.bin to nvidia/ga102/gsp/ucodes-{version}.bin")

# Create a symlink, deleting the existing file/link if necessary
def symlink(dest, source, target_is_directory = False):
    import errno

    try:
        os.symlink(dest, source, target_is_directory = target_is_directory)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(source)
            os.symlink(dest, source, target_is_directory = target_is_directory)
        else:
            raise e

# Create symlinks in the target directory for the other GPUs.  This mirrors
# what the WHENCE file in linux-firmware does.
def symlinks():
    global outputpath
    global version

    print(f"Creating symlinks in {outputpath}/nvidia")
    os.chdir(f"{outputpath}/nvidia")

    for d in ['tu116', 'ga100', 'ad102']:
        os.makedirs(d, exist_ok = True)

    for d in ['tu104', 'tu106']:
        os.makedirs(d, exist_ok = True)
        symlink('../tu102/gsp', f"{d}/gsp", target_is_directory = True)

    os.makedirs('tu117', exist_ok = True)
    symlink('../tu116/gsp', 'tu117/gsp', target_is_directory = True)

    for d in ['ga103', 'ga104', 'ga106', 'ga107']:
        os.makedirs(d, exist_ok = True)
        symlink('../ga102/gsp', f"{d}/gsp", target_is_directory = True)

    for d in ['ad103', 'ad104', 'ad106', 'ad107']:
        # Some older versions of /lib/firmware had symlinks from ad10x/gsp to ad102/gsp,
        # even though there were no other directories in ad10x.  Delete the existing
        # ad10x directory so that we can replace it with a symlink.
        if os.path.islink(f"{d}/gsp"):
            os.remove(f"{d}/gsp")
            os.rmdir(d)
        symlink('ad102', d, target_is_directory = True)

    # TU11x uses the same GSP bootloader as TU10x
    symlink(f"../../tu102/gsp/gen_bootloader.tlv", f"tu116/gsp/gen_bootloader.tlv")

    # TU11x and GA100 use the same generic bootloader as TU10x
    symlink(f"../../tu102/gsp/gen_bootloader.tlv", f"tu116/gsp/gen_bootloader.tlv")
    symlink(f"../../tu102/gsp/gen_bootloader.tlv", f"ga100/gsp/gen_bootloader.tlv")

    # Blackwell is only supported with GSP, so we can symlink the top-level directories
    # instead of just the gsp/ subdirectories.
    for d in ['gb102']:
        symlink('gb100', d, target_is_directory = True)

    for d in ['gb203', 'gb205', 'gb206', 'gb207']:
        symlink('gb202', d, target_is_directory = True)

    # Symlink the GSP-RM image
    symlink(f"../../tu102/gsp/gsp.bin", f"tu116/gsp/gsp.bin")
    symlink(f"../../tu102/gsp/gsp.bin", f"ga100/gsp/gsp.bin")
    symlink(f"../../ga102/gsp/gsp.bin", f"ad102/gsp/gsp.bin")
    symlink(f"../../ga102/gsp/gsp.bin", f"gh100/gsp/gsp.bin")
    symlink(f"../../ga102/gsp/gsp.bin", f"gb100/gsp/gsp.bin")
    symlink(f"../../ga102/gsp/gsp.bin", f"gb202/gsp/gsp.bin")

    # Symlink the ucodes binaries
    if os.path.exists(f"tu102/gsp/ucodes.bin"):
        symlink(f"../../tu102/gsp/ucodes.bin", f"tu116/gsp/ucodes.bin")
        symlink(f"../../tu102/gsp/ucodes.bin", f"ga100/gsp/ucodes.bin")
    if os.path.exists(f"ga102/gsp/ucodes.bin"):
        symlink(f"../../ga102/gsp/ucodes.bin", f"ad102/gsp/ucodes.bin")
        symlink(f"../../ga102/gsp/ucodes.bin", f"gh100/gsp/ucodes.bin")
        symlink(f"../../ga102/gsp/ucodes.bin", f"gb100/gsp/ucodes.bin")
        symlink(f"../../ga102/gsp/ucodes.bin", f"gb202/gsp/ucodes.bin")

def main():
    global outputpath
    global version

    parser = argparse.ArgumentParser(
        description = 'Extract firmware binaries from the OpenRM git repository'
        ' in a format expected by the Nova device drivers.',
        epilog = 'Running as root and specifying -o /lib/firmware will install'
        ' the firmware files directly where Nova expects them.'
        ' The --revision option is useful for testing new firmware'
        ' versions without changing Nova source code.'
        ' The --driver option accepts a .run file path, a URL, or a local'
        ' build output directory.  If -d is given with no argument, the .run'
        ' file is downloaded automatically.')
    parser.add_argument('-i', '--input', default = os.getcwd(),
        help = 'Path to source directory (where version.mk exists)')
    parser.add_argument('-o', '--output', default = os.path.join(os.getcwd(), '_out'),
        help = 'Path to target directory (where files will be written)')
    parser.add_argument('-r', '--revision',
        help = 'Files will be named with this version number')
    parser.add_argument('--debug-fused', action='store_true',
        help = 'Extract debug instead of production images')
    parser.add_argument('-d', '--driver',
        nargs = '?', const = '',
        help = 'Also extract GSP-RM firmware from a source.'
        ' A URL or path to a .run driver package downloads or extracts it.'
        ' A path to a local build output directory (e.g.'
        ' drivers/resman/build/gsp/_out/Linux_amd64_release) copies'
        ' the GSP firmware directly.  If -d is given with no argument,'
        ' the .run file is downloaded automatically.')
    parser.add_argument('-s', '--symlink', action='store_true',
        help = 'Also create symlinks for all supported GPUs')
    parser.add_argument('-w', '--whence', action='store_true',
        help = 'Also generate a WHENCE file')

    args = parser.parse_args()

    args.output = os.path.abspath(args.output)
    if args.driver is not None and args.driver != '' and not re.search('^http[s]://', args.driver):
        args.driver = os.path.abspath(args.driver)

    args.input = os.path.abspath(args.input)
    os.chdir(args.input)

    if not os.path.isfile("version.mk"):
        raise MyException(f"Source directory {args.input} is incorrect")

    version = args.revision
    if not version:
        with open("version.mk") as f:
            version = re.search(r'^NVIDIA_VERSION = ([^\s]+)', f.read(), re.MULTILINE).group(1)
        del f

    if not version.isascii():
        raise MyException(f"Version string {version} must not contain non-ASCII characters")

    print(f"Generating files for version {version}")

    outputpath = args.output;
    print(f"Writing files to {outputpath}")

    os.makedirs(f"{outputpath}/nvidia", exist_ok = True)

    # TU10x and GA100 do not have debug-fused versions of the GSP bootloader
    if args.debug_fused:
        print("Generating images for debug-fused GPUs")
        fuse = "dbg"
        fmc_fuse = "Debug"
    else:
        fuse = "prod"
        fmc_fuse = "Prod"

    # The generic bootloader is only defined for TU102 but is used
    # by all TU1xx and GA100.
    generic_bootloader("tu102")

    booter("tu102", "load", 16, fuse)
    booter("tu102", "unload", 16, fuse)
    gsp_bootloader("tu102")

    booter("tu116", "load", 16, fuse)
    booter("tu116", "unload", 16, fuse)
    # TU11x uses the same bootloader as TU10x

    booter("ga100", "load", 384, fuse)
    booter("ga100", "unload", 384, fuse)
    gsp_bootloader("ga100")

    booter("ga102", "load", 384, fuse)
    booter("ga102", "unload", 384, fuse)
    gsp_bootloader("ga102", fuse)

    booter("ad102", "load", 384, fuse)
    booter("ad102", "unload", 384, fuse)
    gsp_bootloader("ad102", fuse)
    scrubber("ad102", 384, fuse) # Not currently used by Nouveau

    gsp_bootloader("gh100", fuse)
    fmc("gh100", fmc_fuse)

    gsp_bootloader("gb100", fuse)
    fmc("gb100", fmc_fuse)

#    gsp_bootloader("gb10b", fuse)
#    fmc("gb10b", fmc_fuse)

    gsp_bootloader("gb202", fuse)
    fmc("gb202", fmc_fuse)

    gsp_origin = None

    if args.driver is not None:
        if args.driver == '':
            # No path/url provided, so make a guess of the URL
            # to automatically download the right version.
            args.driver = f'https://download.nvidia.com/XFree86/Linux-x86_64/{version}/NVIDIA-Linux-x86_64-{version}.run'

        if re.search('^http[s]://', args.driver):
            with tempfile.NamedTemporaryFile(prefix = f'NVIDIA-Linux-x86_64-{version}-', suffix = '.run') as f:
                print(f"Downloading driver from {args.driver} as {f.name}")
                urllib.request.urlretrieve(args.driver, f.name)
                gsp_firmware_from_run(f.name)
            del f
        elif os.path.isdir(args.driver):
            gsp_firmware_from_build(args.driver)
            gsp_origin = f"local build ({args.driver})"
        else:
            if not os.path.exists(args.driver):
                raise MyException(f"File {args.driver} does not exist.")

            gsp_firmware_from_run(args.driver)

    if args.symlink:
        symlinks()

    if args.whence:
        whence(gsp_origin)

if __name__ == "__main__":
    try:
        main()
    except MyException as e:
        # The full stack trace is too noisy with MyException
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
