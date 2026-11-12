import numpy as np
import matplotlib.pyplot as plt

# This script might be deprecated in the future, as it is a simple example of how to plot a mesh. The main plotting functionality is now integrated into the main codebase, and this script serves as a standalone example.

# get the directory of the current script
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
print(script_dir)


# Load data
vertices = np.loadtxt(os.path.join(script_dir, 'sphere_vertices.csv'), delimiter=',', skiprows=1)
faces = np.loadtxt(os.path.join(script_dir, 'sphere_faces.csv'), delimiter=',', skiprows=1, dtype=int)

# Extract coordinates
x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]

# Create figure
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the mesh
ax.plot_trisurf(x, y, z, triangles=faces, edgecolor='k', linewidth=0.5, alpha=0.8, color='skyblue')

# Formatting (Hide axes for a cleaner look in the thesis)
ax.axis('off')

plt.savefig(os.path.join(script_dir, 'sphere_mesh.png'), dpi=300, bbox_inches='tight', transparent=True)
print("Mesh plot saved to " + os.path.join(script_dir, 'sphere_mesh.png'))
