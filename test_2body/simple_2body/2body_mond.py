"""
two body simulation with 10^10 (2e41 kg) , 10^11  (2e42 kg) solar mass, seperated by 1 Megga parsec. (MOND)
"""
from pathlib import Path
import pysco
import numpy as np
import pandas as pd


# Matter density fraction (Ωₘ)
#rho_object = (2e41 + 2e42) / 1e6  # in kg/Mpc³, assume 2*2*2 Mpc³ volume 
#H0 = 70 * 1e3 / 3.086e22  # Hubble constant in s⁻¹ (70 km/s/Mpc) 
#rho_crit = 3.0 * H0**2 / (8 * np.pi * 6.67430e-11) * (3.086e22)**3 # in kg/Mpc³
#Om_m = rho_object / rho_crit # 8.13e-12

path = Path(__file__).parent.absolute()

param = {
    "nthreads": 1,                    # Use 1 CPU thread. (Can be >1 if you want parallelism.)
    "theory": "mond",
    "mond_function": "simple",       # or "standard"
    "mond_g0": 1.2e-10,              # In m/s², or code units depending on config
    "mond_scale_factor_exponent": 0,
    "mond_alpha": 1,
    "H0": 72,                      # km/s/Mpc
    "Om_m": 0.25733,                  # Matter density fraction (Ωₘ)
    "T_cmb": 2.726,                   # CMB temperature in Kelvin — defines radiation energy density.
    "N_eff": 3.044,                   # Effective number of relativistic species (neutrinos). Usually not used in 2-body sims.
    "w0": -1.0,
    "wa": 0.0,                        # ΛCDMmodel

    # Simulation dimension
    "boxlen": 2,                      # in Mpc/h
    "ncoarse": 2,                     # Mesh resolution = number of grid cells per dimension. Total number of cells = 2**(3*ncoarse)
    "npart": 2,        
    
    # Initial conditions
    "z_start": 10e-3,                     # Initial redshift for the simulation
    "seed": 42,             
    "position_ICS": "center",            # Initial particle positions: "center" or "random"
    "initial_conditions": 00000,                    
    #"power_spectrum_file": f"{path}/pk_lcdmw7v2.dat",                 # Irrelevant here since not generating Gaussian ICs

    # Outputs
    "base": f"{path}/output_newton",
    "output_snapshot_format": "parquet",
    #"z_out": "[0.8, 0.6, 0.4, 0.2, 0.]",
    "z_out": "[5e-3, 4e-3, 3e-3, 2e-3, 0.]",
    "save_power_spectrum": "no",

    # Particles
    "integrator": "leapfrog",               # Integration scheme for time-stepping "Leapfrog" or "Euler"
    "mass_scheme": "TSC",                   # TSC: mass is spread to the 27 nearest grid points (3×3×3 cube) -- kernel to assign particle mass across nearby grid cells
    "n_reorder": 50,                        # Re-order particles every n_reorder steps 
    "mass": [10e10, 10e11],                 # in solar masses
    
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