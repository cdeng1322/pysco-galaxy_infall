import struct

def list_gadget2_blocks(filename):
    with open(filename, "rb") as f:
        f.seek(4 + 256 + 4)  # skip header

        print("Block list in:", filename)
        while True:
            block_header = f.read(4)
            if len(block_header) < 4:
                break  # EOF

            block_header_val = struct.unpack("I", block_header)[0]
            block_name = f.read(4).decode("utf-8")
            block_footer_val = struct.unpack("I", f.read(4))[0]

            data_size = struct.unpack("I", f.read(4))[0]
            f.seek(data_size, 1)  # skip the block data
            data_end = struct.unpack("I", f.read(4))[0]

            print(f"Block: {block_name.strip()} — size: {data_size} bytes")

list_gadget2_blocks("testing/snapshot_000")
