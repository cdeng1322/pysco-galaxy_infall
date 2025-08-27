import pandas as pd
import numpy as np

def find_com(pos, vel, m1, m2, boxsize=10.): # convert to CoM frame
    """
    Transform 2-body positions and velocities into the CoM frame,
    then shift so the barycenter is at the center of the simulation box.

    Parameters
    ----------
    pos : ndarray shape (2, 3)
        Positions of the two bodies in any frame [kpc]
    vel : ndarray shape (2, 3)
        Velocities of the two bodies in any frame [km/s]
    m1, m2 : float
        Masses of the two bodies [same units]
    boxsize : float
        Physical size of the simulation box [Mpc]

    Returns
    -------
    pos_box : ndarray shape (2, 3)
        Positions in the CoM frame shifted to box center [kpc]
    vel_com : ndarray shape (2, 3)
        Velocities in the CoM frame [km/s]
    """
    Mtot = m1+m2
    r_com = (m1*pos[0] + m2*pos[1]) / Mtot
    v_com = (m1*vel[0] + m2*vel[1]) / Mtot
    pos_com = pos - r_com # shift to CoM frame
    vel_com = vel - v_com
    box_center = 0.1 * boxsize * 1000.0
    box_center_vec = np.array([box_center]*3)  # Compute box center vector
    pos_box = pos_com + box_center_vec # Shift CoM to box center
    return pos_box, vel_com


# define position in kpc
# define velocities in km/s
positions = np.array([
    [0., 0., 0.],       # Particle 1 position (kpc)
    [-378., 612., -283.]       # Particle 2 position
], dtype=np.float32)

velocities = np.array([
    [0., 0., 0.],     # Particle 1 velocity (km/s)
    [60.4, -82.7, 54.2]      # Particle 2 velocity
], dtype=np.float32)


m_MW = 0.7e11 # Msun
m_M31 = 1.6e11 # Msun
boxsize = 10 # Mpc
pos, vel = find_com(positions, velocities, m_MW, m_M31, boxsize)
print('MW pos:\n', pos[0])
print('M31 pos:\n', pos[1])
print('MW vel:\n', vel[0])
print('M31 vel:\n', vel[1])

# Define DataFrame with separate x, y, z columns
df = pd.DataFrame({
    "x": pos[:, 0],
    "y": pos[:, 1],
    "z": pos[:, 2],
    "vx": vel[:, 0],
    "vy": vel[:, 1],
    "vz": vel[:, 2]
})

# Save to Parquet
path = "test_2body/galaxy_infall/mond/output_infall_0.1/output_00000/"
df.to_parquet(f"{path}particles_mond_g0_1.2_exponent_0_simple_multigrid_ncoarse5.parquet",index=False)
#df.to_parquet(f"{path}particles_newton_multigrid_ncoarse5.parquet", index=False)

print(f"✅ Parquet file fixed and saved to {path}")
