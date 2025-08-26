import numpy as np
import matplotlib.pyplot as plt


def cartesian_to_polar_hamiltonian(x, y, vx, vy, m=1.0):
    """
    Convert Cartesian coordinates and velocities to polar Hamiltonian variables.

    Parameters:
    x (float): x-coordinate of the particle
    y (float): y-coordinate of the particle
    vx (float): x-component of velocity
    vy (float): y-component of velocity
    m (float): mass of the particle (default=1.0)

    Returns:
    tuple: (q, p) where q = theta (angular coordinate) and p = p_theta (conjugate momentum)
    """
    r = np.sqrt(x ** 2 + y ** 2)
    if r == 0:
        raise ValueError("r cannot be zero")
    theta = np.arctan2(y, x)
    theta_dot = (x * vy - y * vx) / (r ** 2)
    p_theta = m * r ** 2 * theta_dot
    return theta, p_theta


def visualize_trajectory(x, y, vx, vy):
    """
    Visualize a particle's position and velocity in Cartesian and polar coordinates.

    Parameters:
    x (float): x-coordinate
    y (float): y-coordinate
    vx (float): x-velocity
    vy (float): y-velocity
    """
    # Compute polar coordinates
    theta, p_theta = cartesian_to_polar_hamiltonian(x, y, vx, vy)
    r = np.sqrt(x ** 2 + y ** 2)

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot origin
    ax.plot(0, 0, 'ko', label='Origin')

    # Plot particle position
    ax.plot(x, y, 'ro', label='Particle')

    # Plot radial line (representing theta)
    ax.plot([0, x], [0, y], 'b--', label=f'θ = {theta:.2f} rad')

    # Plot velocity vector (scaled for visibility)
    scale = 0.5  # Scale velocity arrow for clarity
    ax.arrow(x, y, vx * scale, vy * scale, color='g', width=0.05,
             head_width=0.2, label='Velocity')

    # Add text for p_theta
    ax.text(0.05, 0.95, f'p_θ = {p_theta:.2f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top')

    # Set equal aspect ratio for proper visualization
    ax.set_aspect('equal')

    # Labels and grid
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)
    ax.legend()
    ax.set_title('Particle Trajectory: Cartesian to Polar Coordinates')

    # Set limits
    max_range = max(np.abs([x, y, x + vx, y + vy])) * 1.5
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)

    plt.show()


# Example usage with sample data
x, y = 1.0, 1.0  # Particle at (1, 1)
vx, vy = -0.5, 1.0  # Velocity vector
visualize_trajectory(x, y, vx, vy)