# FILE: shapexformplot.py
# DATE: 1 JAN 2019
# AUTH: G. E. Deschaines
# DESC: Support module for Jupyter notebook shape_xform.ipynb and
#       executable program to demonstrate plotting 3D parametric
#       shapes using Matplotlib.

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import atan

RPD = atan(1.0)/45.0
DPR = 1.0/RPD

import sys
import traceback

# Robotics Toolbox for Python
# Copyright (C) 1993-2013, by Peter I. Corke
#
# This file is part of The Robotics Toolbox for MATLAB (RTB).
# 
# RTB is free software: you can redistribute it and/or modify 
# it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation, either version
# 3 of the License, or (at your option) any later version.
# 
# RTB is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
# the GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General
# Public License along with RTB. If not, copies may be viewed
# and downloaded at <https://www.gnu.org/licenses/>.
#
# http://www.petercorke.com

# This file incorporates routines extracted from RTB for Python 
# scripts as denoted below. 


###
### Routines extracted from robotics-toolbox-python/robot/utility.py
###

def error(s):
    """
    Common error handler. Display the error string, execute a traceback then raise
    an execption to return to the interactive prompt.
    """
    print("Robotics toolbox error: %s" % s)

    #traceback.print_exc()
    raise ValueError()


def numcols(m):
    """
    Number of columns in a matrix.

    @type m: matrix
    @return: the number of columns in the matrix.
    return m.shape[1]
    """
    return m.shape[1]


def numrows(m):
    """
    Number of rows in a matrix.

    @type m: matrix
    @return: the number of rows in the matrix.
    return m.shape[1]
    """
    return m.shape[0]


def ishomog(tr):
    """
    True if C{tr} is a 4x4 homogeneous transform.

    @note: Only the dimensions are tested, not whether the rotation
    submatrix is orthonormal.

    @rtype: boolean
    """
    if tr is None:
        return False
    try:
        size = tr.shape
        return size == (4,4)
    except AttributeError:
        return False


###
### Routines extracted from robotics-toolbox-python/robot/transform.py
###

def rotx(theta):
    """
    Rotation about X-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about X-axis

    @see: L{roty}, L{rotz}, L{rotvec}
    """
    ct = np.cos(theta)
    st = np.sin(theta)

    return mat([[1,  0,    0],
                [0,  ct, -st],
                [0,  st,  ct]])


def roty(theta):
    """
    Rotation about Y-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about Y-axis

    @see: L{rotx}, L{rotz}, L{rotvec}
    """
    ct = np.cos(theta)
    st = np.sin(theta)

    return np.asmatrix([[ct,   0,   st],
                        [0,    1,    0],
                        [-st,  0,   ct]])


def rotz(theta):
    """
    Rotation about Z-axis
    
    @type theta: number
    @param theta: the rotation angle
    @rtype: 3x3 orthonormal matrix
    @return: rotation about Z-axis

    @see: L{rotx}, L{roty}, L{rotvec}
    """
    ct = np.cos(theta)
    st = np.sin(theta)

    return np.asmatrix([[ct,      -st,  0],
                        [st,       ct,  0],
                        [ 0,        0,  1]])


def r2t(R):
    """
    Convert a 3x3 orthonormal rotation matrix to a 4x4 homogeneous transformation::
    
        T = | R 0 |
            | 0 1 |
            
    @type R: 3x3 orthonormal rotation matrix
    @param R: the rotation matrix to convert
    @rtype: 4x4 homogeneous matrix
    @return: homogeneous equivalent
    """    
    return np.concatenate( (np.concatenate( (R, np.zeros((3,1))),1), np.asmatrix([0,0,0,1])) )


def eul2r(phi, theta=None, psi=None):
    """
    Rotation from Euler angles.
    
    Two call forms:
        - R = eul2r(S{theta}, S{phi}, S{psi})
        - R = eul2r([S{theta}, S{phi}, S{psi}])
    These correspond to rotations about the Z, Y, Z axes respectively.

    @type phi: number or list/array/matrix of angles
    @param phi: the first Euler angle, or a list/array/matrix of angles
    @type theta: number
    @param theta: the second Euler angle
    @type psi: number
    @param psi: the third Euler angle
    @rtype: 3x3 orthonormal matrix
    @return: R([S{theta} S{phi} S{psi}])

    @see:  L{tr2eul}, L{eul2tr}, L{tr2rpy}
    """
    n = 1
    if theta == None and psi==None:
        # list/array/matrix argument
        phi = np.mat(phi)
        if numcols(phi) != 3:
            error('bad arguments')
        else:
            n = numrows(phi)
            psi = phi[:,2]
            theta = phi[:,1]
            phi = phi[:,0]
    elif (theta!=None and psi==None) or (theta==None and psi!=None):
        error('bad arguments')
    elif not isinstance(phi,(int,np.int32,float,np.float64)):
        # all args are vectors
        phi = np.mat(phi)
        n = numrows(phi)
        theta = np.mat(theta)
        psi = np.mat(psi)

    if n>1:
        R = []
        for i in range(0,n):
            r = rotz(phi[i,0]) * roty(theta[i,0]) * rotz(psi[i,0])
            R.append(r)
        return R
    try:
        r = rotz(phi[0,0]) * roty(theta[0,0]) * rotz(psi[0,0])
        return r
    except:
        r = rotz(phi) * roty(theta) * rotz(psi)
        return r


def eul2tr(phi,theta=None,psi=None):
    """
    Rotation from Euler angles.
    
    Two call forms:
        - R = eul2tr(S{theta}, S{phi}, S{psi})
        - R = eul2tr([S{theta}, S{phi}, S{psi}])
    These correspond to rotations about the Z, Y, Z axes respectively.

    @type phi: number or list/array/matrix of angles
    @param phi: the first Euler angle, or a list/array/matrix of angles
    @type theta: number
    @param theta: the second Euler angle
    @type psi: number
    @param psi: the third Euler angle
    @rtype: 4x4 homogenous matrix
    @return: R([S{theta} S{phi} S{psi}])

    @see:  L{tr2eul}, L{eul2r}, L{tr2rpy}
    """
    return r2t( eul2r(phi, theta, psi) )


def transl(x, y=None, z=None):
    """
    Create or decompose translational homogeneous transformations.
    
    Create a homogeneous transformation
    ===================================
    
    - T = transl(v)
    - T = transl(vx, vy, vz)
        
    The transformation is created with a unit rotation submatrix.
    The translational elements are set from elements of v which is
    a list, array or matrix, or from separate passed elements.
    
    Decompose a homogeneous transformation
    ======================================
    
    v = transl(T)   
    
    Return the translation vector
    """
           
    if y == None and z == None:
        x = np.mat(x)
        try:
            if ishomog(x):
                return x[0:3,3].reshape(3,1)
            else:
                return np.concatenate((np.concatenate((np.eye(3),x.reshape(3,1)),1),np.mat([0,0,0,1])))
        except AttributeError:
            n = len(x)
            r = [[],[],[]]
            for i in range(n):
                r = np.concatenate((r,x[i][0:3,3]),1)
            return r
    elif y != None and z != None:
        return np.concatenate((np.concatenate((eye(3),np.mat([x,y,z]).T),1),np.mat([0,0,0,1])))


###
### Parametric shape coordinate definition and plotting routines
###

def parametric_frame(s):
    """ Parametric Cartesian coordinate frame
    """
    x = s * np.asmatrix([[0.0, 0.0, 0.0],[1.0, 0.0, 0.0],[0.0, 0.0, 0.0]])
    y = s * np.asmatrix([[0.0, 0.0, 0.0],[0.0, 1.0, 0.0],[0.0, 0.0, 0.0]])
    z = s * np.asmatrix([[0.0, 0.0, 0.0],[0.0, 0.0, 1.0],[0.0, 0.0, 0.0]])
    return (x, y, z)


def parametric_box(s):
    """ Parametric box shape
    """
    r = s*np.sqrt(2.0)/2
    h = s/2
    u = np.linspace(0.25*np.pi, 2.25*np.pi, 5)
    v = np.linspace(-1.0, 1.0, 5)
    x = r * np.outer(np.cos(u), np.ones(np.size(v)))
    y = r * np.outer(np.sin(u), np.ones(np.size(v)))
    z = h * np.outer(np.ones(np.size(u)), v)
    return (x, y, z)


def parametric_sphere(d, dim):
    """ Parametric sphere shape
    """
    r = d/2
    u = np.linspace(0.0, 2*np.pi, dim)
    v = np.linspace(0.0, np.pi, dim)
    x = r * np.outer(np.cos(u), np.sin(v))
    y = r * np.outer(np.sin(u), np.sin(v))
    z = r * np.outer(np.ones(np.size(u)), np.cos(v))
    return (x, y, z)


def parametric_cylinder(d, l, dim):
    """ Parametric cylinder shape
    """
    r = d/2
    h = l/2
    u = np.linspace(0.0, 2*np.pi, dim)
    v = np.linspace(-1.0, 1.0, dim)
    x = r * np.outer(np.cos(u), np.ones(np.size(v)))
    y = r * np.outer(np.sin(u), np.ones(np.size(v)))
    z = h * np.outer(np.ones(np.size(u)), v)
    return (x, y, z)


def parametric_cone(d0, d1, l, dim):
    """ Parametric cone shape
    """
    r0 = d0/2.0
    r1 = d1/2.0
    f  = (r1-r0)/2.0
    h  = l/2
    u = np.linspace(0.0, 2*np.pi, dim)
    v = np.linspace(-1.0, 1.0, dim)
    s = r0 + f*(v+1.0)
    x = s * np.outer(np.cos(u), np.ones(np.size(v)))
    y = s * np.outer(np.sin(u), np.ones(np.size(v)))
    z = h * np.outer(np.ones(np.size(u)), v)
    return (x, y, z)


def parametric_disk(d, h, dim):
    """ Parametric disk shape
    """
    r = d/2
    u = np.linspace(0.0, 2*np.pi, dim)
    v = np.linspace(0.0, 1.0, dim)
    x = r * np.outer(np.cos(u), v)
    y = r * np.outer(np.sin(u), v)
    z = h * np.outer(np.ones(np.size(u)), np.ones(np.size(v)))
    return (x, y, z)


def parametric_plane(s, h, dim):
    """ Parametric plane shape
    """
    r = s/2
    u = np.linspace(-1.0, 1.0, dim)
    v = np.linspace(-1.0, 1.0, dim)
    x = r * np.outer(u, np.ones(np.size(v)))
    y = r * np.outer(np.ones(np.size(u)), v)
    z = h * np.outer(np.ones(np.size(u)), np.ones(np.size(v)))
    return (x, y, z)


def shape_xform(x, y, z, Tr):
    """ Shape coordinates transformation
    """
    # get dimensions of parametric space (assumed square)
    dim = x.shape[0]

    # pack homogeneous shape coordinates
    xyz1 = np.vstack([x.reshape((1,dim*dim)), 
                      y.reshape((1,dim*dim)),
                      z.reshape((1,dim*dim)),
                      np.ones((1,dim*dim))])

    # apply transform to packed shape coordinates
    Vtr = np.dot(Tr, xyz1)
    """ NumPy matrix type work with Matplotlib's plot routines, but not ipyvolume's.
    xr = Vtr[0,:].reshape((dim,dim))
    yr = Vtr[1,:].reshape((dim,dim))
    zr = Vtr[2,:].reshape((dim,dim))
    """
    xr = np.asarray(Vtr[0,:].reshape((dim,dim)))
    yr = np.asarray(Vtr[1,:].reshape((dim,dim)))
    zr = np.asarray(Vtr[2,:].reshape((dim,dim)))

    return (xr, yr, zr)


shapes = ['frame', 'box', 'sphere', 'cylinder', 'cone', 'disk', 'plane']

def plot_parametric_shape(shape, solid=False, Tr=np.eye(4), **opts):
    """ Plot specified parametric shape
    """
    global shapes

    if shape == 'frame':
        dim = 3
    elif shape == 'box':
        dim = 5
    elif shape in ['sphere', 'cylinder', 'cone', 'disk', 'plane']:
        dim = 100
    else:
        print("*** error: invalid specified shape %s" % shape)
        print("           must be %s" % shapes)
        return

    # create parametric shape coordinates
    if shape == 'frame':
        s = 1.0
        if 's' in opts: s = opts['s']
        (x, y, z) = parametric_frame(s)
    elif shape == 'box':
        s = 1.0
        if 's' in opts: s = opts['s']
        (x, y, z) = parametric_box(s)
    elif shape == 'sphere':
        d = 1.0
        if 'd' in opts: d = opts['d']
        (x, y, z) = parametric_sphere(d, dim)
    elif shape == 'cylinder':
        d = 1.0
        if 'd' in opts: d = opts['d']
        l = 1.0
        if 'l' in opts: l = opts['l']
        (x, y, z) = parametric_cylinder(d, l, dim)
    elif shape == 'cone':
        d0 = 1.0
        if 'd0' in opts: d0 = opts['d0']
        d1 = 0.5
        if 'd1' in opts: d1 = opts['d1']
        l  = 1.0
        if 'l' in opts: l = opts['l']
        (x, y, z) = parametric_cone(d0, d1, l, dim)
    elif shape == 'disk':
        d = 1.0
        if 'd' in opts: d = opts['d']
        h = 0.0
        if 'h' in opts: h = opts['h']
        (x, y, z) = parametric_disk(d, h, dim)
    elif shape == 'plane':
        s = 1.0
        if 's' in opts: s = opts['s']
        h = 0.0
        if 'h' in opts: h = opts['h']
        (x, y, z) = parametric_plane(s, h, dim)

    (xr, yr, zr) = shape_xform(x, y, z, Tr)

    ax = plt.gca()

    if shape == 'frame':
        tail = 0
        head = 1
        ax.quiver(xr[tail,:], yr[tail,:], zr[tail,:],
                  xr[head,:], yr[head,:], zr[head,:], 
                  arrow_length_ratio=0.1, normalize=False, color='k')
        ax.text3D(xr[head,0], yr[head,0], zr[head,0], 'X', ha='left', va='center', color='r')
        ax.text3D(xr[head,1], yr[head,1], zr[head,1], 'Y', ha='left', va='center', color='g')
        ax.text3D(xr[head,2], yr[head,2], zr[head,2], 'Z', ha='left', va='center', color='b')
    elif shape == 'box':
        if solid:
           ax.plot_surface(xr, yr, zr, rstride=1, cstride=5, color='m')
        else:
           ax.plot_wireframe(xr, yr, zr, rstride=1, cstride=5, color='m')
    else:
        if solid:
           ax.plot_surface(xr, yr, zr, rstride=4, cstride=4, color='m')
        else:
           ax.plot_wireframe(xr, yr, zr, rstride=4, cstride=4, color='m')


if __name__ == '__main__':

    shape = 'box'  # a parametric shape
    solid = False  # shape a solid?
    phi   = 0.0    # shape's roll about Z-axis
    theta = 0.0    # shape's -pitch about Y-axis
    psi   = 0.0    # shape's yaw about Z-axis

    # decode command line arguments

    if ( len(sys.argv)  < 2 ) :
        print("")
        print('usage: python3 shapexformplot.py shape solid "{key:val pairs, ...}"')
        print('where: shape is one of %s.' % shapes)
        print('       solid is T or F indicating shape rendered as surface (T) or mesh (F).')
        print('       "{key:val pairs, ...}" is a set of parametric shape parameter options.\n')
        print("e.g.:  python3 shapexformplot.py cylinder F \"{'d':1.0,'l':2.0,'phi':45}\"\n")
        print("       generates a magenta colored mesh cylinder 1 unit in diameter and")
        print("       2 units in length rotated 45 degrees around it's Z-axis.\n")
        sys.exit()

    opts = {}
    if ( len(sys.argv) > 1 ) :
       shape = sys.argv[1]
       if shape not in shapes:
           print("*** error: invalid specified shape %s" % shape)
           print("           must be %s" % shapes)
           sys.exit()
    if ( len(sys.argv) > 2 ) :
       if sys.argv[2][0].upper() == 'T' : solid = True
    if ( len(sys.argv) > 3 ) :
       opts = eval(sys.argv[3])
       if 'phi' in opts: phi = opts['phi']
       if 'theta' in opts: theta = opts['theta']
       if 'psi' in opts: psi = opts['psi']

    # create homogeneous transform
    Tr = eul2tr(phi*RPD, theta*RPD, psi*RPD)

    # create plot figure window
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect("equal")
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    limits = [-2.0, 2.0]
    ax.set_xlim(limits); ax.set_ylim(limits); ax.set_zlim(limits)

    # plot specified parametric figure
    plot_parametric_shape(shape, solid, Tr, **opts)
    plot_parametric_shape('frame', None, Tr, s=2.0)
    plt.show()

