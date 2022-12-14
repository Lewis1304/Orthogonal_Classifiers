from ncon import ncon
import numpy as np
from numpy import eye, zeros
import matplotlib.pyplot as plt
from sample_data import vonMisesFisherSphere, plotPointsSphere
from scipy.linalg import polar

def init_V(D):
    """ Returns a random unitary """
    A = np.random.rand(D,D)
    V,x = np.linalg.qr(A)
    return V

def get_s(d):
    """ returns a spin up vector useful later"""
    s1=np.zeros((d),dtype=np.float64)
    s1[0]=1.0
    s2=np.zeros((d),dtype=np.float64)
    s2[1]=1.0
    return s1,s2

def get_Polar(M):
    """ Return the polar decomposition of M """
    x,y,z =  np.linalg.svd(M)
    print("    S: {}".format(y))
#    print("    U: {}".format(x))
#    print("    V: {}".format(z))
    M = ncon([x,z],([-1,1],[1,-2]))

    return M

def get_Data(Theta, phi, chi, dx, dy, angles=False, coords=False):
    """ Returns a set of spinors on the block shere """
    """ i. centred at theta,phi """
    """ ii. from sterographic projection of gaussian variance dx,dy"""
    """ iii. rotated to an angle chi in plane"""
    N = 1000
    X = np.random.normal(0.0,dx,N)
    Y = np.random.normal(0.0,dy,N)
    Thet = 2*np.arctan2(2, (np.sqrt(X*X+Y*Y)))
    Ph = np.arctan2(Y, X)+chi
    Z = zeros((N,2),dtype=complex)
    m = 1
    while m < N+1:
        Z[m-1,0] =  np.cos(Thet[m-1]/2)+1j*0.0
        Z[m-1,1] = np.sin(Thet[m-1]/2)*np.exp(1j*Ph[m-1])
        m +=1
    if angles:
        Angles = zeros((N, 2))
        Angles[:, 0] = Thet
        Angles[:, 1] = Ph
        return Z, Angles
    if coords:
        Coords = zeros((N, 3))
        Coords[:, 0] = np.sin(Thet)*np.cos(Ph)
        Coords[:, 1] = np.sin(Thet)*np.sin(Ph)
        Coords[:, 2] = np.cos(Thet)
        return Z, Coords
    return Z

def get_Classify1(U,D1,D2):
    N = 1000
    s1,s2 = get_s(2)
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    """ prob of right for 1 minus prob of wrong """
    c1 = np.real(ncon([np.conj(D1),np.conj(U),Z,U,D1]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/N
    """ prob of right for 1 minus prob of wrong """
    c2 = np.real(ncon([np.conj(D2),np.conj(U),Z,U,D2]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/N
    Cost = c1-c2
    """ Now calculate an equivalent of the training accuracy """
    l=1
    acc1 = 0.0
    acc2 = 0.0
    while l < N+1:
        D1temp = D1[l-1,:]
        D2temp = D2[l-1,:]
        acc1 = acc1 + (1.0+np.sign(np.real(ncon([np.conj(D1temp),\
        np.conj(U),Z,U,D1temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        acc2 = acc2 -(-1.0+np.sign(np.real(ncon([np.conj(D2temp),\
        np.conj(U),Z,U,D2temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        l+=1
    Accuracy = (acc1+acc2)/(4*N)
    return Cost,Accuracy


def get_ImproveU1(U,D1,D2, alpha=0.5):
    """ This attempts to improve the stacking using polar decomp"""
    """ for a single copy nothing changes """
    from copy import copy
    N = 1000

    Z = eye(2)
    Z[1,1] = - Z[1,1]
    m = 1
    DU1 = applyU1(D1, U)
    DU2 = applyU1(D2, U)
    ax = plotBlochSphere(DU1, spinToBloch=True)
    ax = plotBlochSphere(DU2, spinToBloch=True, ax=ax, c='r',
            title='Not Improved m={}'.format(m))
    print("    {}".format(get_Classify1(U,D1,D2)))

    print(D1.shape)
    print('Trying to improve...')
    _U = copy(U)
    dist = U - _U
    dist_sq = dist @ dist.conj().T
    print('Orig dist from original U: {}'.format(np.trace(dist_sq)))
    while m < 15:
        # Taking the trace is equivalent to summing over the images
#        dz1 = ncon([np.conj(D1),Z,U,D1]\
#        ,([5,-2],[3,-1],[3,4],[5,4]))

        mydz1 = ncon([Z, U, D1, np.conj(D1)], ((-1, 1), (1, 2), (3, 2), (3, -2)))

#        dz2 = ncon([np.conj(D2),Z,U,D2]\
#        ,([5,-2],[3,-1],[3,4],[5,4]))

        mydz2 = ncon([Z, U, D2, np.conj(D2)], ((-1, 1), (1, 2), (3, 2), (3, -2)))

        U = get_Polar(U + alpha*(mydz1-mydz2))
#        U = get_Polar(dz1 - dz2)

        dist = U - _U
        dist_sq = dist @ dist.conj().T
        print('    Trace dist from original U: {}'.format(np.trace(dist_sq)))

        m +=1

        DU1 = applyU1(D1, U)
        DU2 = applyU1(D2, U)

#        ax = plotBlochSphere(DU1, spinToBloch=True)
#        ax = plotBlochSphere(DU2, spinToBloch=True, ax=ax, c='r',
#                title='Improved m={}'.format(m))
        cost, acc = get_Classify1(U, D1, D2)
        print("    m = {}: Cost = {}, Acc = {}".format(m, cost, acc))
        print()
    print(get_Classify1(U,D1,D2))
#    plt.show()
    return U

def applyU1(D1, U):
    states = [ncon([U, d1], ((-1, 1), (1,))) for d1 in D1]
    states = np.array(states)
    return states

def get_Classify2(Z2,FF1,FF2):
    N = 1000
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    ZI = ncon([Z,eye(2)],([-1,-3],[-2,-4])).reshape([2*2,2*2])
    """ prob of right for 1 minus prob of wrong """
    z1 = np.real(ncon([np.conj(FF1),np.conj(Z2),ZI,Z2,FF1]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/N
    """ prob of right for 1 minus prob of wrong """
    z2 = np.real(ncon([np.conj(FF2),np.conj(Z2),ZI,Z2,FF2]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/N
    Cost = z1-z2
    """ Now calculate an equivalent of the training accuracy """
    l=1
    acc1 = 0.0
    acc2 = 0.0
    while l < N+1:
        FF1temp = FF1[l-1,:]
        FF2temp = FF2[l-1,:]
        acc1 = acc1 + (1.0+np.sign(np.real(ncon([np.conj(FF1temp),np.conj(Z2),\
        ZI,Z2,FF1temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        acc2 = acc2 -(-1.0+np.sign(np.real(ncon([np.conj(FF2temp),np.conj(Z2),\
        ZI,Z2,FF2temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        l+=1
    Accuracy = (acc1+acc2)/(4*N)
    return Cost, Accuracy

def get_Z2(U,D1,D2):
    """ This attempts to improve the stacking using polar decomp"""
    """ for a single copy nothing changes """
    N = 1000
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    II = eye(4)
    ZI = ncon([Z,eye(2)],([-1,-3],[-2,-4])).reshape([2*2,2*2])
    """ initialise Z2 to identity"""
    """Z2 = init_V(4)"""
    Z2 = eye(4)
    """ Apply improved feature selector then copy data """
    F1 = ncon([U,D1],([-2,1],[-1,1]))
    F2 = ncon([U,D2],([-2,1],[-1,1]))
    FF1 = np.zeros([N,4],dtype=complex)
    FF2 = np.zeros([N,4],dtype=complex)
    l = 1
    while l < N+1:
        tempF1 =  F1[l-1,:]
        tempF2 =  F2[l-1,:]
        FF1[l-1,:] = ncon([tempF1,tempF1],([-1],[-2])).reshape([2*2])
        FF2[l-1,:] = ncon([tempF2,tempF2],([-1],[-2])).reshape([2*2])
        l +=1
    """ construct circuits for derivative of cost function and polarise"""
    print('Pre optimisation accuracy + cost:')
    print(get_Classify2(Z2,FF1,FF2))
    print("Attempting optimisation")
    m=1
    while m < 6:
        dz1 = ncon([np.conj(FF1),(ZI+II),Z2,FF1]\
        ,([5,-2],[-1,3],[3,4],[5,4]))
        dz2 = ncon([np.conj(FF2),(ZI-II),Z2,FF2]\
        ,([5,-2],[-1,3],[3,4],[5,4]))
        Z2 = get_Polar(dz1-dz2)
        cost, acc = get_Classify2(Z2, FF1, FF2)
        print("   m: {}  |  Cost : {}  |  Acc : {}".format(m, cost, acc))
        m +=1
    print(get_Classify2(Z2,FF1,FF2))
    return Z2

def get_Classify3(Z3,FFF1,FFF2):
    N = 1000
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    ZII = ncon([Z,eye(2),eye(2)],([-1,-4],[-2,-5],[-3,-6])).reshape([2*2*2,2*2*2])
    """ prob of right for 1 minus prob of wrong """
    c1 = np.real(ncon([np.conj(FFF1),np.conj(Z3),ZII,Z3,FFF1]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/N
    """ prob of right for 1 minus prob of wrong """
    c2 = np.real(ncon([np.conj(FFF2),np.conj(Z3),ZII,Z3,FFF2]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/N
    Cost = c1-c2
    """ Now calculate an equivalent of the training accuracy """
    l=1
    acc1 = 0.0
    acc2 = 0.0
    while l < N+1:
        FFF1temp = FFF1[l-1,:]
        FFF2temp = FFF2[l-1,:]
        acc1 = acc1 + (1.0+np.sign(np.real(ncon([np.conj(FFF1temp),np.conj(Z3),\
        ZII,Z3,FFF1temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        acc2 = acc2 -(-1.0+np.sign(np.real(ncon([np.conj(FFF2temp),np.conj(Z3),\
        ZII,Z3,FFF2temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        l+=1
    Accuracy = (acc1+acc2)/(4*N)
    return Cost,Accuracy

def get_Z3(U,Z2,D1,D2):
    """ This attempts to improve the stacking using polar decomp"""
    """ for a single copy nothing changes """
    N = 1000
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    III = eye(8)
    ZII = ncon([Z,eye(2),eye(2)],([-1,-4],[-2,-5],[-3,-6])).reshape([2*2*2,2*2*2])
    """ initialise Z3 to Z2*I"""
    """Z3 = eye(8)"""
    Z3 = ncon([Z2,eye(2)],([-1,-3],[-2,-4])).reshape([4*2,4*2])
    """ Apply improved feature selector then copy data """
    F1 = ncon([U,D1],([-2,1],[-1,1]))
    F2 = ncon([U,D2],([-2,1],[-1,1]))
    FFF1 = np.zeros([N,8],dtype=complex)
    FFF2 = np.zeros([N,8],dtype=complex)
    l = 1
    while l < N+1:
        tempF1 =  F1[l-1,:]
        tempF2 =  F2[l-1,:]
        FFF1[l-1,:] = ncon([tempF1,tempF1,tempF1],([-1],[-2],[-3])).reshape([2*2*2])
        FFF2[l-1,:] = ncon([tempF2,tempF2,tempF2],([-1],[-2],[-3])).reshape([2*2*2])
        l +=1
    """print(get_Classify3(Z3,FFF1,FFF2))"""
    """ construct circuits for derivative of cost function and polarise"""
    m=1
    while m < 20:
        dz1 = ncon([np.conj(FFF1),(ZII+III),Z3,FFF1]\
        ,([5,-2],[-1,3],[3,4],[5,4]))
        dz2 = ncon([np.conj(FFF2),(ZII-III),Z3,FFF2]\
        ,([5,-2],[-1,3],[3,4],[5,4]))
        Z3 = get_Polar(dz1-dz2)
        """print(get_Classify3(Z3,FFF1,FFF2))"""
        m +=1
    print(get_Classify3(Z3,FFF1,FFF2))
    return Z3

def get_Classify4(Z4,FFFF1,FFFF2):
    N = 1000
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    ZIII = ncon([Z,eye(2),eye(2),eye(2)],([-1,-5],[-2,-6],[-3,-7],[-4,-8])).reshape([2*2*2*2,2*2*2*2])
    """ prob of right for 1 minus prob of wrong """
    c1 = np.real(ncon([np.conj(FFFF1),np.conj(Z4),ZIII,Z4,FFFF1]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/N
    """ prob of right for 1 minus prob of wrong """
    c2 = np.real(ncon([np.conj(FFFF2),np.conj(Z4),ZIII,Z4,FFFF2]\
    ,([5,1],[2,1],[3,2],[3,4],[5,4])))/N
    Cost = c1-c2
    """ Now calculate an equivalent of the training accuracy """
    l=1
    acc1 = 0.0
    acc2 = 0.0
    while l < N+1:
        FFFF1temp = FFFF1[l-1,:]
        FFFF2temp = FFFF2[l-1,:]
        acc1 = acc1 + (1.0+np.sign(np.real(ncon([np.conj(FFFF1temp),np.conj(Z4),\
        ZIII,Z4,FFFF1temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        acc2 = acc2 -(-1.0+np.sign(np.real(ncon([np.conj(FFFF2temp),np.conj(Z4),\
        ZIII,Z4,FFFF2temp]\
        ,([1],[2,1],[3,2],[3,4],[4])))))
        l+=1
    Accuracy = (acc1+acc2)/(4*N)
    return Cost,Accuracy

def get_Z4(U,Z3,D1,D2):
    """ This attempts to improve the stacking using polar decomp"""
    """ for a single copy nothing changes """
    N = 1000
    Z = eye(2)
    Z[1,1] = - Z[1,1]
    IIII = eye(16)
    ZIII = ncon([Z,eye(2),eye(2),eye(2)],([-1,-5],[-2,-6],[-3,-7],[-4,-8])).reshape([2*2*2*2,2*2*2*2])
    """ initialise Z4 to Z3*I"""
    Z4 = ncon([Z3,eye(2)],([-1,-3],[-2,-4])).reshape([8*2,8*2])
    """ Apply improved feature selector then copy data """
    F1 = ncon([U,D1],([-2,1],[-1,1]))
    F2 = ncon([U,D2],([-2,1],[-1,1]))
    FFFF1 = np.zeros([N,16],dtype=complex)
    FFFF2 = np.zeros([N,16],dtype=complex)
    l = 1
    while l < N+1:
        tempF1 =  F1[l-1,:]
        tempF2 =  F2[l-1,:]
        FFFF1[l-1,:] = ncon([tempF1,tempF1,tempF1,tempF1],([-1],[-2],[-3],[-4]))\
        .reshape([2*2*2*2])
        FFFF2[l-1,:] = ncon([tempF2,tempF2,tempF2,tempF2],([-1],[-2],[-3],[-4]))\
        .reshape([2*2*2*2])
        l +=1
    """ print(get_Classify4(Z4,FFFF1,FFFF2)) """
    """ construct circuits for derivative of cost function and polarise"""
    m=1
    while m < 20:
        dz1 = ncon([np.conj(FFFF1),(ZIII+IIII),Z4,FFFF1]\
        ,([5,-2],[-1,3],[3,4],[5,4]))
        dz2 = ncon([np.conj(FFFF2),(ZIII-IIII),Z4,FFFF2]\
        ,([5,-2],[-1,3],[3,4],[5,4]))
        Z4 = get_Polar(dz1-dz2)
        """print(get_Classify4(Z4,FFFF1,FFFF2))"""
        m +=1
    print(get_Classify4(Z4,FFFF1,FFFF2))
    return Z4

def get_Features(D1,D2):
    """ construct the classifier initialisation from Z1 and Z2 at order n"""
    N = 1000
    s1,s2 = get_s(2)
    aveD1 = np.sum(D1,axis=0)
    aveD1 = aveD1/(np.sqrt(ncon([aveD1,np.conj(aveD1)],([1],[1]))))
    aveD2 = np.sum(D2,axis=0)
    aveD2 = aveD2/(np.sqrt(ncon([aveD2,np.conj(aveD2)],([1],[1]))))
    U = ncon([s1,np.conj(aveD1)],([-1],[-2]))\
    +ncon([s2,np.conj(aveD2)],([-1],[-2]))
    U = get_Polar(U)
    return U

def plotBlochSphere(states, show=False, ax=None, c='b', title=None,
        spinToBloch=False):
    '''
    Take in state data and ploit on the block sphere
    '''

    if spinToBloch:
        states = [spin_to_bloch(s) for s in states]
        states = np.array(states)
    xs = states[:, 0]
    ys = states[:, 1]
    zs = states[:, 2]

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
    #plt.savefig('toy_stacking_results/sphere_n_total_500_n_test_02_sigma_06_seed_42.pdf')
    if show:
        plt.show()

    return ax

def spin_to_bloch(state):
    u = state[1] /(state[0] + 1e-12)
    ux = np.real(u)
    uy = np.imag(u)

    Z = 1 + ux**2 + uy**2

    x = 2 * ux / Z
    y = 2 * uy / Z
    z = (1-ux**2-uy**2) / Z

    return [x, y, z]

def bloch_to_spin(state):
    r = np.linalg.norm(state)
    phi = np.arctan2(state[1] , state[0])
    theta = np.arccos(state[2] / r)

    if phi < 0.0:
        phi = 2*np.pi + phi

    Z = np.zeros(2, complex)
    Z[0] = np.cos(theta/2.)
    Z[1] = np.sin(theta/2.) * np.exp(1j*phi)

    return Z


def spherical_coords(theta_phi):
    r = 1.0
    theta, phi = theta_phi[0], theta_phi[1]
    return np.array([r*np.sin(theta)*np.cos(phi),r*np.sin(theta)*np.sin(phi),r*np.cos(theta)]).T


if __name__ == "__main__":
    from scipy.stats import unitary_group
    #np.random.seed(1)
    N = 1000
    # Stereographic data generation - TO FIX
    Theta1 = 0.0
    phi1 = 0.1
#    chi1 = 0.0
#    dx1 = 0.1
#    dy1 = 0.1
    Theta2 = 1.0
    phi2 = 0.9
#    chi2 = 1.0
#    dx2 = 0.3
#    dy2 = 0.1
#
#    _, D1 = get_Data(Theta1,phi1,chi1,dx1,dy1, coords=True)
#    _, D2 = get_Data(Theta2,phi2,chi2,dx2,dy2, coords=True)
#
#    ax = plotBlochSphere(D1, spinToBloch=False)
#    ax = plotBlochSphere(D2, spinToBloch=False, show=False, ax=ax, c='r',
#            title='Stereographic distribution')
#
    μ1 = spherical_coords([Theta1, phi1])
    k1 = 10
    μ2 = spherical_coords([Theta2, phi2])
    k2 = 40
    D1 = vonMisesFisherSphere(μ1, k1, N)
    D2 = vonMisesFisherSphere(μ2, k2, N)

#    ax = plotBlochSphere(D1, spinToBloch=False)
#    ax = plotBlochSphere(D2, spinToBloch=False, show=False, ax=ax, c='r',
#            title='vonMises distribution')
    D1 = np.array([bloch_to_spin(d) for d in D1])
    D2 = np.array([bloch_to_spin(d) for d in D2])

    ax = plotBlochSphere(D1, spinToBloch=True)
    ax = plotBlochSphere(D2, spinToBloch=True, show=False, ax=ax, c='r',
            title='v2nMises distribution to spin')

    U = get_Features(D1,D2)
#    U = unitary_group.rvs(U.shape[0])
    C = get_Classify1(U,D1,D2)
    print(C)
#    print("Improved U, 1 copy")
#    Uimproved = get_ImproveU1(U,D1,D2)
#    C = get_Classify1(Uimproved,D1,D2)
    print("Improved 2 copies")
    Z2 = get_Z2(U,D1,D2)
    assert()
    print("Improved 3 copies")
    Z3 = get_Z3(U,Z2,D1,D2)
    print("Improved 4 copies")
    Z4 = get_Z4(U,Z3,D1,D2)

    plt.show()
