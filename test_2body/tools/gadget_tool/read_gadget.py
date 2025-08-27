import pandas as pd
import numpy as np
import numpy.typing as npt
import logging
from typing import Tuple
from pysco import utils


def read_gadget(
    param: pd.Series,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Read initial conditions from Gadget snapshot

    The snapshot can be divided in multiple files, such as \\
    snapshot_X.Y, where Y is the file number. \\
    In this case only keep snapshot_X

    Parameters
    ----------
    param : pd.Series
        Parameter container

    Returns
    -------
    Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]
        Position, Velocity [3, Npart]

    Example
    -------
    >>> from pysco.initial_conditions import read_hdf5
    >>> param = pd.Series({
    ...     'initial_conditions': "file.h5",
    ...     'npart': 128**3,
    ...  })
    >>> position, velocity = read_gadget(param)
    """

    from pysco import readgadget  # From Pylians

    logging.warning(f"Read {param['initial_conditions']}")
    filename = param["initial_conditions"]
    ptype = 1  # DM particles
    header = readgadget.header(filename)
    BoxSize = header.boxsize/1e3 # in Mpc/h
    Nall = header.nall  # Total number of particles
    Omega_m = header.omega_m
    Omega_l = header.omega_l
    h = header.hubble
    redshift = header.redshift  # redshift of the snapshot
    aexp = 1.0 / (1 + redshift)
    param["aexp"] = aexp
    param["z_start"] = 1.0 / aexp - 1
    logging.warning(f"Initial redshift snapshot at z = {1./param['aexp'] - 1}")
    utils.set_units(param)

    npart = int(Nall[ptype])
    if npart != param["npart"]:
        raise ValueError(f"{npart=} and {param['npart']} should be equal.")
    if not np.allclose([Omega_m, 100 * h], [param["Om_m"], param["H0"]]):
        raise ValueError(
            f"Cosmology mismatch: {Omega_m=} {param['Om_m']=} {(100*h)=} {param['H0']=}"
        )
    
    # read positions, velocities of the particles

    print("READING POS block...")
    position = readgadget.read_block(filename, "POS ", [ptype])/1e3 #positions in Mpc/h
    print("READING VEL block...")
    velocity = readgadget.read_block(filename, "VEL ", [ptype])
    #print("READING ID block...")
    #ids = readgadget.read_block(filename, "ID  ", [ptype])-1   #IDs starting from 0

    vel_factor = param["unit_t"] / param["unit_l"]
    utils.prod_vector_scalar_inplace(position, np.float32(1.0 / BoxSize))
    utils.prod_vector_scalar_inplace(velocity, np.float32(vel_factor))
    return position, velocity

