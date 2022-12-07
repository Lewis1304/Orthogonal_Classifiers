# Generate sample data by sampling von Mises-Fisher distribution on 2 sphere.
#
# Inspired by https://dlwhittenbury.github.io/ds-2-sampling-and-visualising-the-von-mises-fisher-distribution-in-p-dimensions.html
#

import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt

def sample_uniform_on_sphere(N, d=2):
    samples = np.random.randn(N, d)

    norms = np.linalg.norm(samples, axis=1)

    samples = (samples.T / norms).T

    return samples

def sample_t_marginal_distribution(κ, N):
    # Note in this p is set to 3
    p = 3.
    maxiters = 10000
    b = (p-1) / (2*κ + np.sqrt(4*κ**2 + (p-1)**2))

    x0 = (1-b) / (1+b)
    c = κ*x0 + (p-1) * np.log(1 - x0**2)

    samples = np.zeros((N, 1))

    for i in range(N):
        iters = 0

        # Rejection sampling to find W
        while iters < maxiters:
            Z = np.random.beta( (p-1)/2, (p-1)/2)
            U = np.random.rand(1)
            W = (1-(1+b)*Z)/(1-(1-b)*Z)

            if κ*W + (p-1)*np.log(1-x0*W) -c >= np.log(U): # Acceptance criteria for W
                samples[i] = W
                break

            iters += 1
    return samples

def vonMisesFisherSphere(μ, κ, N):

    μ = μ / np.linalg.norm(μ) # Force μ to be a unit vector.

    samples = np.zeros((N, 3)) # Empty array to store samples

    # Calculate components orthogonal to μ
    xi = sample_uniform_on_sphere(N)

    # Components in the direction of μ
    t = sample_t_marginal_distribution(κ, N)

    samples[:, 0] = t.squeeze()

    samples[:, 1:] =  np.sqrt(1-t**2) * xi

    # Rotate in direction of μ
    μ = np.reshape(μ, (len(μ), 1))
    O = null_space(μ.T)
    R = np.concatenate((μ, O), axis=1)
    samples = np.dot(R, samples.T).T

    return samples


def plotPointsSphere(points, show=False, ax=None, c='b', title=None,
        marker='o'):
    '''
    Take in state data and ploit on the block sphere
    '''
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

    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    ax.scatter(xs, ys, zs, marker=marker, s=25, edgecolor='k', c=c)

    ax.set_xlim([-1,1])
    ax.set_ylim([-1,1])
    ax.set_zlim([-1,1])
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()

    if show:
        plt.show()

    return ax


if __name__=="__main__":
    μ = np.random.rand(3)
    μ = μ / np.linalg.norm(μ)
    κ = 20
    N = 100
    samples = vonMisesFisherSphere(μ, κ, N)

    ax = plotPointsSphere(samples, show=False)
    plotPointsSphere(np.array([μ]), show=True, ax=ax, c='red', marker="^")





