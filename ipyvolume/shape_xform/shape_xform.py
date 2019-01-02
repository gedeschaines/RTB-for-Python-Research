
# coding: utf-8

# In[1]:


# FILE: shape_xform.ipynb
# DATE: 1 JAN 2019
# AUTH: G. E. Deschaines
# DESC: This Jupyter notebook presents a rudimentary implementation of the basic Robotics Toolbox (RTB) for MATLAB
#       3D graphic elements -- coordinate frame transform and manipulator arm/joint cylinder -- displayed by the
#       trplot() function and SerialLink.plot() method respectively. The implementation presented herein is only a
#       proof of principle demonstration for suitability of ipyvolume/ipywidgets as a RTB for Python GUI supported 
#       by Jupyter lab and notebooks.


# In[2]:


import ipyvolume.pylab as p3
import ipyvolume as ipv
import numpy as np

from shapexformplot import *  # provides RTB for Python components to create and manipulate 3D graphic elements


# In[3]:


# define a parametric cylinder and a coordinate frame about the world space origin
(Xco, Yco, Zco) = parametric_cylinder(1.0, 2.0, 32)
(Xfo, Yfo, Zfo) = parametric_frame(1)


# In[4]:


# create homogeneous transform
phi = -45.0                          # shape's roll about Z-axis
theta = np.linspace(0.0, 360.0, 36)  # shape's -pitch about Y-axis
psi = 0.0                            # shape's yaw about Z-axis

Tr = []
k = len(theta)
for i in range(0,k):
   tr =  eul2tr(phi*RPD, theta[i]*RPD, psi*RPD)
   Tr.append(tr)


# In[5]:


# apply transform to the parametric shape coordinates
Xc = np.ndarray( (k, Xco.shape[0], Xco.shape[1]) );
Yc = np.ndarray( (k, Yco.shape[0], Yco.shape[1]) );
Zc = np.ndarray( (k, Zco.shape[0], Zco.shape[1]) );
Xf = np.ndarray( (k, Xfo.shape[0], Xfo.shape[1]) );
Yf = np.ndarray( (k, Yfo.shape[0], Yfo.shape[1]) );
Zf = np.ndarray( (k, Zfo.shape[0], Zfo.shape[1]) );
for i in range(0,k):
    (Xc[i,:,:], Yc[i,:,:], Zc[i,:,:]) = shape_xform(Xco, Yco, Zco, Tr[i])
    (Xf[i,:,:], Yf[i,:,:], Zf[i,:,:]) = shape_xform(Xfo, Yfo, Zfo, Tr[i])


# In[6]:


# create ipyvolume figure to display cylinder and coordinate frame transform animation
fig = p3.figure(width=480, height=480)

# set ipyvolume style properties
p3.style.background_color('black')
p3.style.box_off()
p3.style.use("dark")
p3.style.use({'xaxis.color':"red"})    # <-+
p3.style.use({'yaxis.color':"green"})  # <-+- do not appear to be working
p3.style.use({'zaxis.color':"blue"})   # <-+

# set figure view and axes 
p3.view(0.0, -120.0)
p3.xyzlim(-2.0, 2.0)
p3.xyzlabel('X','Y','Z')

# coordinate frame axes line segments
Lx = p3.plot(Xf[:,0:2,0], Yf[:,0:2,0], Zf[:,0:2,0], color='red', size=2)
Lx.material.visible = False
Lx.line_material.visible = True
Ly = p3.plot(Xf[:,0:2,1], Yf[:,0:2,1], Zf[:,0:2,1], color='green')
Ly.material.visible = False
Ly.line_material.visible = True
Lz = p3.plot(Xf[:,0:2,2], Yf[:,0:2,2], Zf[:,0:2,2], color='blue')
Lz.material.visible = False
Lz.line_material.visible = True

# coordinate frame axes line segment tip arrows
Xfv = Xf[:,1,:] - Xf[:,0,:]
Yfv = Yf[:,1,:] - Yf[:,0,:]
Zfv = 3*(Zf[:,1,:] - Zf[:,0,:])
arrowcols = np.zeros((k,3,3))
arrowcols[0:k,0,0] = 1.0
arrowcols[0:k,1,1] = 1.0
arrowcols[0:k,2,2] = 1.0
q = p3.quiver(Xf[:,1,:], Yf[:,1,:], Zf[:,1,:], Xfv[:,:], Yfv[:,:], Zfv[:,:],
              size=10, size_selected=5, color=arrowcols, color_selected='gray')

# cylinder body surface
s = p3.plot_surface(Xc, Yc, Zc, color='orange')

# pass ipyvolume objects to animation controller and show
p3.animation_control([Lx,Ly,Lz,q,s], interval=100)
p3.show()


# In[7]:


p3.save("shape_xform.html",offline=True)

