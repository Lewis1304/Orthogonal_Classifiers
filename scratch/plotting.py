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


def plotBlochSphere(states, show=False, ax=None, colour='b', title=None,
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
        ax = prepareBlochSphereAxes()

    ax.scatter(xs, ys, zs, marker='o', s=25, edgecolor='k', c=colour)

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

def prepareBlochSphereAxes():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    r=1
    phi, theta = np.mgrid[0.0:np.pi:100j, 0.0:2.0*np.pi:100j]
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    ax.plot_wireframe(
        x, y, z,  rstride=1, cstride=1, cmap=plt.cm.YlGnBu_r, alpha=0.8, linewidth=0.5)
    return ax


def plotGreatCircle(p1, p2, ax=None, colour='k', alpha=0.5):
    '''
    Plot a great circle on a unit sphere given two points on the great circle
    `p1` and `p2`.
    '''
    points = greatCirclePoints(p1, p2)
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    if ax is None:
        ax = prepareBlochSphereAxes()
    ax.plot(xs, ys, zs=zs, c=colour, alpha=alpha)
    return ax

def plotVector(point, ax=None, origin=None, colour='k', alpha=1.0, normalise=False):
    if ax is None:
        ax = prepareBlochSphereAxes()
    if origin is None:
        origin = np.array([0., 0., 0.])

    if normalise:
        point = point / np.linalg.norm(point)
    points = np.zeros((2, 3))
    points[0, :] = origin
    points[1, :] = point
    xs, ys, zs = points[:, 0], points[:, 1], points[:, 2]
    ax.plot(xs, ys, zs=zs, c=colour, alpha=alpha)
    return ax


def greatCirclePoints(p1, p2, N=100):
    '''
    From two arbitrary points `p1` and `p2`, compute N equidistant points on
    a great circle of a unit sphere.
    '''
    p1 = p1 / np.linalg.norm(p1)
    p2 = p2 / np.linalg.norm(p2)

    # Compute two orthogonal vectors on great circle
    u = p1
    w = np.cross(p1, p2)
    w = w / np.linalg.norm(w)
    v = np.cross(u, w)

    # Parameterise angles
    N = 100
    ω = np.linspace(0, 2*np.pi, N)
    ω = np.repeat(ω, 3)
    ω = ω.reshape(N, 3)

    #Compute points of great circle
    circle = u*np.cos(ω) + v*np.sin(ω)

    return circle

if __name__=="__main__":
    # Exampled plotting of N random states
    N = 100
    states1 = np.random.randn(N, 2) + 1j*np.random.randn(N, 2)
    # Normalise the states
    norms = np.linalg.norm(states1, axis=1)
    states1 = states1 / np.array([norms, norms]).T


    # Generate another set of random states
    states2 = np.random.randn(N, 2) + 1j*np.random.randn(N, 2)
    # Normalise the states
    norms = np.linalg.norm(states2, axis=1)
    states2 = states2 / np.array([norms, norms]).T

    # plot states in different colours on Bloch sphere
    ax = plotBlochSphere(states1, spinToBloch=True, colour='green')
    ax = plotBlochSphere(states2, spinToBloch=True, colour='blue', ax=ax)

    # Plot a random great circle
    p1 = np.random.rand(3)
    p2 = np.random.rand(3)

    ax = plotGreatCircle(p1, p2, ax)

    # Plot the great circle vectors
    ax = plotVector(p1, normalise=True, ax=ax, colour='r')
    ax = plotVector(p2, normalise=True, ax=ax, colour='r')

    plt.show()
