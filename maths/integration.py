import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Unit conversions
kpc_to_m = 3.085677581e19        # 1 kpc in meters
kmps_to_mps = 1000              # km/s to m/s
Msun_to_kg = 1.98847e30         # solar mass in kg
yr_to_s = 365.25 * 24 * 3600     # 1 year in seconds

# Constants and conversions
G_val = 6.67430e-11  # m^3 kg^-1 s^-2
G_val = G_val / (kpc_to_m)**3 * (yr_to_s*1e6)**2  # kpc^3 kg^-1 Myr^-2
a0 = 1.2e-10  # m/s^2
a0 = a0 / kpc_to_m * (yr_to_s*1e6)**2  # kpc Myr^-2

# Physical parameters
m1 = 0.7e11 * Msun_to_kg      # kg
m2 = 1.6e11 * Msun_to_kg      # kg
# Q_val = 1.0                     # assume Q ~ 1 for now
q1 = m1/(m1+m2)  
q2 = 1 - q1  
Q = 2*(1-q1**1.5-q2**1.5)/(3*q1*q2)
vconv = 1.022e-3 # multiply by vconv to convert velocity from km/s to kpc/Myr

## Define function 

def K(t, om_m=1/3): # t = 0 at Big Bang
    H0 = 1/14e3 # in Myr^-1
    om_lambda = 1 - om_m
    return H0**2 * om_lambda - 2/9 * t **(-2)


#K = lambda t: (1./2. * (1/14e9)**2 * (4./3. - 1./3.*scalefac(t)**(-3))) # t=0 at Bigbang

def acceleration(t, rmag, K, m1, m2, F): # calculate acceleration scaler 
    acc = K * rmag - (m1 + m2) / m1 * F / m2
    return acc


def eofm(t, y, t0=14e3): 
    global b, om_m
    """
    This defines the second order ODE for MONDian infall
    t: time; 
    y: state vector [r1_x, r1_y, r1_z, r2.., v1.., v2..] 12 elements
    """
    r1 = y[:3] # position 1
    r2 = y[3:6] # position 2
    v1 = y[6:9] # velocity 1
    v2 = y[9:] # velocity 2
    abs_r12 = np.linalg.norm(r2 - r1)  # distance
    rn = np.sqrt(abs_r12**2 + 2*b**2)   # softened distance
    ev = (r2 - r1) / abs_r12 # unit vector 
    tt = t0 + t
    y_term = (np.sqrt(G_val * (m1 + m2) * a0) / (rn * Q * a0))**2
    F = G_val * m1 * m2 / rn**2 * (1 + y_term**(-0.5))
    acce = acceleration(t, rn, K(tt, om_m=om_m), m1, m2, F) #relative acceleration = a2 - a1
    acc1 = -acce * ev * m2/(m1+m2) # acceleration of r1, follows m1a1 + m2a2 = 0
    acc2 = acce * ev * m1/(m1+m2)
    return np.concatenate((v1, v2, acc1, acc2))

# Initial conditions in CoM: r = 770 kpc, v = 109 km/s, tangential = 17 km/s
r12 = 770
v12 = -109 # (M31 toward MW)
vt  = -17.0  
r1 = np.array([r12, 0., 0.]) * -q1 # MW position 
r2 = np.array([r12 , 0., 0.]) * q2 # M31 position in kpc
v1 = np.array([v12, vt, 0.]) * -q1 * vconv# MW velocity in kpc/Myr
v2 = np.array([v12, vt, 0.]) * q2 * vconv # M31 velocity: toward MW
y0 = np.concatenate((r1, r2, v1, v2)) 

# Time array from now (0) to 14 Gyr ago (negative direction, in seconds)
t_start = 0e3
t_end = -13e3
t_eval = np.linspace(t_start, t_end, 10000)

# set global parameters
b = 1
om_m = 1/3
# Solve the IVP
sol = solve_ivp(eofm, [t_start, t_end], y0, t_eval=t_eval, method='RK45', rtol=1e-6, # allow fractional error ~10e-6
    atol=1e-9, # this dominate when the solution is small, allow absolute error < 10e-9
    max_step=50.0) # never take a step longer than 50 Myr.
time_gyr = sol.t / 1e3
r1 = sol.y[0:3]
r2 = sol.y[3:6]
distance_kpc = np.linalg.norm(r2 - r1, axis=0)
print(f"Minimum separation in the past: {np.min(distance_kpc):.2f} kpc at time {time_gyr[np.argmin(distance_kpc)]:.2f} Gyr ago")

# Plotting
plt.plot(time_gyr, distance_kpc, color='red', lw=1.5, label='Past')
t_start = 0e3
t_end = 20e3
t_eval = np.linspace(t_start, t_end, 1000)
sol = solve_ivp(eofm, [t_start, t_end], y0, t_eval=t_eval, method='RK45', rtol=1e-6,
    atol=1e-9,
    max_step=50.0)
time_gyr = sol.t / 1e3
plt.plot(time_gyr, np.linalg.norm(sol.y[3:6] - sol.y[0:3], axis=0), color='blue', lw=1.5, label='Future')
plt.xlabel("Time (Gyr) (0 = present)")
plt.ylabel("Separation (kpc)")
plt.title(f"MOND universe: Om_m={om_m:.2f}, Om_L={1-om_m:.2f}")
plt.grid(True, alpha=0.5)
plt.legend()
plt.show()
