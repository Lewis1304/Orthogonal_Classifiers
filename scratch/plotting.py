import numpy as np
import matplotlib.pyplot as plt


def spin_to_bloch(state):
    '''
    Convert a spin state `a|0> + b|1>` represented as a list `[a, b]` into
    the Cartesian coordinates for the point on the Bloch sphere.
    '''
    u = state[1] /(state[0] + 1e-12)
    ux = np.real(u)
    uy = np.imag(u)

    Z = 1 + ux**2 + uy**2

    x = 2 * ux / Z
    y = 2 * uy / Z
    z = (1-ux**2-uy**2) / Z

    return [x, y, z]


def plotBlochSphere(states, show=False, ax=None, c='b', title=None,
        spinToBloch=False):
    '''
    Take in state points and plot on the Bloch sphere
    '''

    # Convert spin states to Cartesian data points.
    if spinToBloch:
        states = [spin_to_bloch(s) for s in states]
        states = np.array(states)
    xs = states[:, 0]
    ys = states[:, 1]
    zs = states[:, 2]

    # Set up axes with wirefram sphere
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        r=1
        phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
        x = r*np.sin(phi)*np.cos(theta)
        y = r*np.sin(phi)*np.sin(theta)
        z = r*np.cos(phi)
        ax.plot_wireframe(
            x, y, z,  rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r, alpha=0.8, linewidth=0.5)

    ax.scatter(xs, ys, zs, marker='o', s=25, edgecolor='k', c=c)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    if title is not None:
        ax.set_title(title)
    if show:
        plt.show()

    return ax

if __name__=="__main__":
    # Exampled plotting of N random states
    N = 100
    states = np.random.randn(N, 2) + 1j*np.random.randn(N, 2)
    # Normalise the states
    norms = np.linalg.norm(states, axis=1)
    states = states / np.array([norms, norms]).T

    # plot on Bloch sphere
    plotBlochSphere(states, spinToBloch=True)
    plt.show()
