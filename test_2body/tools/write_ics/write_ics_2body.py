'''
This generates initial condition with 2 particles with 10^10, 10^11 solar mass.
The file is generated in the GADGET-2 binary format, which can be read by Pylians 
'''

import numpy as np
import struct

filename = "testing/snapshot_000"  

# Number of particles of each type: (only type 1 is used here)
npart = [0, 2, 0, 0, 0, 0]
massarr = [0, 0, 0, 0, 0, 0]  # mass per particle type (set to 0 when using MassBlock)

# Particle positions in Mpc/h
pos = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0]  # 1 Mpc apart along x-axis
], dtype=np.float32)

# Velocities in km/s (start at rest)
vel = np.zeros_like(pos, dtype=np.float32)

# IDs
ids = np.array([1, 2], dtype=np.uint32)

# Masses in solar masses
mass = np.array([1e10, 1e11], dtype=np.float32)

# Header data (GADGET-2 format)
header_format = (
    6 * "I"     # npart (6 ints)
    + 6 * "d"   # massarr (6 doubles)
    + "d"       # time
    + "d"       # redshift
    + "i"       # flag_sfr
    + "i"       # flag_feedback
    + 6 * "I"   # npartTotal
    + "i"       # flag_cooling
    + "i"       # num_files
    + "d"       # BoxSize
    + "d"       # Omega0
    + "d"       # OmegaLambda
    + "d"       # HubbleParam
    + "96x"     # fill up to 256 bytes
)
header_data = struct.pack(
    header_format,
    *npart,
    *massarr,
    0.0,  # time
    0.0,  # redshift
    0, 0,  # flags
    *npart,
    0, 1,  # cooling flag, num_files
    100,  # box size in Mpc/h
    0.3, 0.7,  # Omega0, OmegaLambda
    0.7  # HubbleParam
)

def write_block(f, label, data):
    blockname = label.encode("utf-8")
    data = np.asarray(data, dtype=np.float32 if label in ["POS ", "VEL "] else data.dtype)
    f.write(struct.pack("I", 8))       # size of block name section
    f.write(blockname)                 # block name (e.g. b"POS ")
    f.write(struct.pack("I", 8))       # size again
    f.write(struct.pack("I", data.nbytes))  # data block size
    data.tofile(f)
    f.write(struct.pack("I", data.nbytes))  # data block size again


with open(filename, "wb") as f:
    # Header
    f.write(struct.pack("I", 256))
    f.write(header_data)
    f.write(struct.pack("I", 256))

    # POS block
    write_block(f, "POS ", pos)

    # VEL block
    write_block(f, "VEL ", vel)

    # ID block
    write_block(f, "ID  ", ids)

    # MASS block
    write_block(f, "MASS", mass)

