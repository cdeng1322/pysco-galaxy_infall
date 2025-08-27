"""
Simulation of two body with 1e12 Msun.
Initial conditions for galaxy infall: one at rest, one moving at 110 km/s towards the other,
seperated by 770 kpc.
To avoid periodic boundary effects, I need to set boxlen > 1Mpc.

"""
from pathlib import Path
import pysco
import pandas as pd
import numpy as np
from astropy.constants import G, M_sun, pc


def find_boxlen(l_proper, H0=70., aexp=1.):
    """
    Calculate the box length in Mpc/h given Universe proper length in Mpc.
    
    Parameters:
    - l_proper: Universe's proper length in Mpc.
    - H0: Hubble constant in km/s/Mpc (default is 70).
    - aexp: Scale factor (default is 1, present time).
    
    Returns:
    - Box length in Mpc/h (comoving unit).
    """
    boxlen = l_proper / aexp * H0 / 100.0  # Convert to Mpc/h
    if boxlen < 1.0:
        raise ValueError("Box length must be at least 1 Mpc/h to avoid periodic boundary effects.")
    return boxlen


def find_omh_m(size, mass, H0=70.):
    """
    Calculate the matter density fraction (Ωₘ) of universe given its size and mass.
    
    Parameters:
    - size: assume a cubic volume, size is the proper length of one side in Mpc.
    - mass: total mass inside the volume in solar masses.
    
    Returns:
    - Density in Msun/Mpc^3.
    """
    if size <= 0:
        raise ValueError("Each side must be greater than zero.")
    if mass <= 0:
        raise ValueError("Total mass must be greater than zero.")
    
    # compute rho_m
    volume = size ** 3
    rho_m = mass / volume  # Msun/Mpc^3
    # compute rho_crit = 3H_0^2 / 8πG
    Mpc_to_m = (1e6 * pc).value  # Mpc -> m
    H0 = H0 * 1000 / Mpc_to_m  # km/s/Mpc -> 1/s
    rho_crit = 3 * H0**2 / (8*np.pi*G.value) # kg/m³
    rho_crit = rho_crit / M_sun.value * Mpc_to_m**3 # Msun/Mpc^3

    # Ωₘ = rho_m / rho_crit
    omh_m = rho_m / rho_crit

    return omh_m

m_MW = 0.7e11
m_M31 = 1.6e11
print('boxlen = ', find_boxlen(10), 'Mpc/h')
print('omh_m = ',find_omh_m(10, (m_MW+m_M31)))



path = Path(__file__).parent.absolute()

z_out = np.linspace(-0.01, -0.6, 30).tolist()  # Redshifts at which to output snapshots
#z_out = [-0.01, -0.05, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]
param = {
    "nthreads": 1,                    # Use 1 CPU thread. (Can be >1 if you want parallelism.)
    "theory": "newton",               # "newton" or "mond"
    #"mond_function": "simple",       # or "standard"
    #"mond_g0": 1.2e-10,              # In m/s², or code units depending on config
    #"mond_scale_factor_exponent": 0,
    #"mond_alpha": 1,
    "H0": 70,                               # km/s/Mpc
    "Om_m": find_omh_m(10, (m_MW+m_M31)),  # Matter density fraction (Ωₘ)
    "T_cmb": 0.,                            # CMB temperature in Kelvin — defines radiation energy density.
    "N_eff": 0.,                            # Effective number of relativistic species (neutrinos). Usually not used in 2-body sims.
    "w0": 0.0,
    "wa": 0.0,                              # ΛCDMmodel

    # Simulation dimension
    "boxlen": find_boxlen(10),           # in Mpc/h
    "ncoarse": 5,                     # ncells_1d = 2^ncoarse
    "npart": 2,        
    
    # Initial conditions
    "z_start": 0.,                     # Initial redshift for the simulation
    "seed": 42,             
    "position_ICS": "center",            # Initial particle positions: "center" or "random"
    "initial_conditions": 00000,                                    # Irrelevant here since not generating Gaussian ICs
    "write_snapshot": True,          # Write initial snapshot to file

    # Outputs
    "base": f"{path}/output_infall",
    "output_snapshot_format": "parquet",
    #"z_out": "[-0.05, -0.07, -0.08, -0.09, -0.1,-0.12, -0.13, -0.14, -0.15, -0.2, -0.25, -0.3, -0.35, -0.4]",
    "z_out":  str(z_out), 
    "save_power_spectrum": "no",

    # Particles
    "integrator": "leapfrog",               # Integration scheme for time-stepping "Leapfrog" or "Euler"
    "mass_scheme": "TSC",                   # TSC: mass is spread to the 27 nearest grid points (3×3×3 cube) -- kernel to assign particle mass across nearby grid cells
    "n_reorder": 50,                        # Re-order particles every n_reorder steps 
    "mass": [m_MW, m_M31],               # in solar masses
    
    # Time stepping
    "Courant_factor": 1.0,                  # Cell fraction for time stepping based on velocity/acceleration (Courant_factor < 1 means more time steps)
    "max_aexp_stepping": 10,                # Maximum percentage [%] of scale factor that cannot be exceeded by a time step
    
    # Newtonian solver
    "linear_newton_solver": "multigrid",    # Linear solver for Newton's method: "multigrid", "fft", "fft_7pt" or "full_fft"
    "gradient_stencil_order": 5,            # n-point stencil with n = 2, 3, 5 or 7  
    
    # Multigrid
    "Npre": 2,                          # Number of pre-smoothing Gauss-Seidel iterations
    "Npost": 1,                         # Number of post-smoothing Gauss-Seidel iterations
    "epsrel": 1e-2,                     # Maximum relative error on the residual norm
    
    # Verbose
    "verbose": 1                   # Verbose level. 0 : silent, 1 : basic infos, 2 : full timings

}

param["mass"] = pd.Series(param["mass"])


# Run simulation
pysco.run(param)

print("Run completed!")
