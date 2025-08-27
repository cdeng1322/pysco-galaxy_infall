import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os

# Path to simulation outputs
base = "output_mond"

# Number of frames (check how many output_xxxxxx folders exist)
nframes = 7  # adjust to your case

positions = []

for i in range(nframes):
    path = f"testing/{base}/output_{i:05d}/particles_newton_multigrid_ncoarse2.parquet"
    table = pq.read_table(path)
    
    # Extract positions
    x = np.array(table["Position"][:, 0])
    y = np.array(table["Position"][:, 1])
    
    positions.append((x, y))

fig, ax = plt.subplots(figsize=(6, 6))
sc = ax.scatter([], [], s=50)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title("Particle Motion (comoving)")

def update(frame):
    x, y = positions[frame]
    sc.set_offsets(np.column_stack((x, y)))
    ax.set_title(f"Frame {frame}")
    return sc,

ani = animation.FuncAnimation(fig, update, frames=nframes, blit=True)
ani.save("testing/2body_newton_motion.gif", fps=2)
plt.show()
