# Winn Koster  --  Final Project  --  Orbital DEQ proof of concept
# ephemeris: https://ssd.jpl.nasa.gov/horizons.cgi

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

G = 6.67408e-11
M = 1.989e+30
u = G*M
au_distance = 1.496e+11
#earth_distance = 1.49598023e+11
v_i = np.sqrt( u / au_distance )        # v initial is the earth

velocity_grid_points = 100

v_x_i_array = np.linspace(-20.,20., velocity_grid_points)
v_y_i_array = np.linspace(32715.,32755., velocity_grid_points)

r_grid = np.zeros( (velocity_grid_points,velocity_grid_points) )

# in the DEQ below, we have made the following substitution to turn a 2nd order diffeq into 2 coupled first order diffeqs...
# OG equation in radial...
# decomposed into cartesian...              \ddot x \ = \ \frac{-\mu x}{(x^2 + y^2 + z^2)^{3/2}}
# then we set x_dot as the substitution dummy var

def x_pos_2d(x_vec,y_vec,z_vec,t):                     # x_vec[] is an array that encodes the x-location x[0] and the x-velocity x[1]. We've called this 'v_x', so that we may maintain using 'd_' as a prefix (eg: d_x and d_v_x) in order to be consistent with other codes
    x = x_vec[0]
    y = y_vec[0]
    z = z_vec[0]
    v_x = x_vec[1]
    d_x = v_x
    d_v_x = -(u*x)/(( (x**2.) + (y**2.) + (z**2.) )**(3./2.) )
    return np.array([d_x,d_v_x],float)

def y_pos_2d(x_vec,y_vec,z_vec,t):                     # x_vec[] is an array that encodes the x-location x[0] and the x-velocity x[1]. We've called this 'v_x', so that we may maintain using 'd_' as a prefix (eg: d_x and d_v_x) in order to be consistent with other codes
    x = x_vec[0]
    y = y_vec[0]
    z = z_vec[0]
    v_y = y_vec[1]
    d_y = v_y
    d_v_y = -(u*y)/(( (x**2.) + (y**2.) + (z**2.) )**(3./2.) )
    return np.array([d_y,d_v_y],float)

def x_pos(x_vec,y_vec,z_vec,t):                     # x_vec[] is an array that encodes the x-location x[0] and the x-velocity x[1]. We've called this 'v_x', so that we may maintain using 'd_' as a prefix (eg: d_x and d_v_x) in order to be consistent with other codes
    x = x_vec[0]
    y = y_vec[0]
    z = z_vec[0]
    v_x = x_vec[1]
    d_x = v_x
    #d_v_x = -(u*x* (((x**2.)+(y**2.))**(1./2.)) )/(( (x**2.) + (y**2.) + (z**2.) )**(2.) )
    d_v_x = -(u*x)/(( (x**2.) + (y**2.) + (z**2.) )**(3./2.) )
    return np.array([d_x,d_v_x],float)

def y_pos(x_vec,y_vec,z_vec,t):                     # x_vec[] is an array that encodes the x-location x[0] and the x-velocity x[1]. We've called this 'v_x', so that we may maintain using 'd_' as a prefix (eg: d_x and d_v_x) in order to be consistent with other codes
    x = x_vec[0]
    y = y_vec[0]
    z = z_vec[0]
    v_y = y_vec[1]
    d_y = v_y
    #d_v_y = -(u*y* (((x**2.)+(y**2.))**(1./2.)) )/(( (x**2.) + (y**2.) + (z**2.) )**(2.) )
    d_v_y = -(u*y)/(( (x**2.) + (y**2.) + (z**2.) )**(3./2.) )

    return np.array([d_y,d_v_y],float)

def z_pos(x_vec,y_vec,z_vec,t):                     # x_vec[] is an array that encodes the x-location x[0] and the x-velocity x[1]. We've called this 'v_x', so that we may maintain using 'd_' as a prefix (eg: d_x and d_v_x) in order to be consistent with other codes
    x = x_vec[0]
    y = y_vec[0]
    z = z_vec[0]
    v_z = z_vec[1]
    d_z = v_z
    d_v_z = -(u*z)/(( (x**2.) + (y**2.) + (z**2.) )**(3./2.) )
    return np.array([d_z,d_v_z],float)

t_0 = 0.0
t_1 = 22368649.              # Hohmann transfer Earth-Mars time elapsed
#N = 10000                               # for fast runs
N = 40000                               # medium length
#N = 2000000                             # setting this higher than 1000000 does NOT help the one year simulation at all. It's initial conditions at that point...
h = (t_1-t_0)/N

print(' ')
print(str(N)+' points in a simulation that lasts for '+str(t_1/31536000.)+' years.')
print('Points sampled using RK4 every '+str(h)+' seconds, or '+str(h/3600.)+' hours.')
print(' ')

i=0
while i < len(v_y_i_array):
    j=0
    while j < len(v_x_i_array):

        print('i = '+str(i)+', j = '+str(j))
        # Begin numerical simulation...
        tpoints = np.arange(t_0,t_1,h)
        x_pos_points = []                   # x[0], the x position
        x_vel_points = []                   # x[1], the x velocity
        y_pos_points = []                   # y[0], the y position
        y_vel_points = []                   # y[1], the y velocity
        z_pos_points = []                   # z[0], the z position
        z_vel_points = []                   # z[1], the z velocity

        # initial conditions
        x_vec = np.array([au_distance, v_x_i_array[j] ],float)         # initial [x_pos,x_vel], will be updated upon iteration
        y_vec = np.array([0., v_y_i_array[i] ],float)         # initial [y_pos,y_vel], will be updated upon iteration
        z_vec = np.array([0.,0.],float)         # initial [z_pos,z_vel], will be (eventually) updated upon iteration (when we add z variance)

        for t in tpoints:
            x_pos_points.append(x_vec[0])
            x_vel_points.append(x_vec[1])
            y_pos_points.append(y_vec[0])
            y_vel_points.append(y_vec[1])
            z_pos_points.append(z_vec[0])
            z_vel_points.append(z_vec[1])

            kx1 = h*x_pos(x_vec, y_vec, z_vec, t)
            kx2 = h*x_pos(x_vec+0.5*kx1, y_vec, z_vec, t+0.5*h)
            kx3 = h*x_pos(x_vec+0.5*kx2, y_vec, z_vec, t+0.5*h)
            kx4 = h*x_pos(x_vec+kx3, y_vec, z_vec, t+h)

            ky1 = h*y_pos(x_vec, y_vec, z_vec, t)
            ky2 = h*y_pos(x_vec, y_vec+0.5*ky1, z_vec, t+0.5*h)
            ky3 = h*y_pos(x_vec, y_vec+0.5*ky2, z_vec, t+0.5*h)
            ky4 = h*y_pos(x_vec, y_vec+ky3, z_vec, t+h)

            # Uncomment these when you actually hahve a z_pos() function...
            kz1 = h*z_pos(x_vec, y_vec, z_vec, t)
            kz2 = h*z_pos(x_vec, y_vec, z_vec+0.5*kz1, t+0.5*h)
            kz3 = h*z_pos(x_vec, y_vec, z_vec+0.5*kz2, t+0.5*h)
            kz4 = h*z_pos(x_vec, y_vec, z_vec+kz3, t+h)

            # We've verified that altering the order of these updates doens't change anything in the sim. Not that we think it would, but just checking. We've 'baked' all the data for the new sim in the previous lines already...
            x_vec += (kx1+2.*kx2+2.*kx3+kx4)/6.
            z_vec += (kz1+2.*kz2+2.*kz3+kz4)/6.
            y_vec += (ky1+2.*ky2+2.*ky3+ky4)/6.

            # radius calculation
            radius_error = np.sqrt( (x_pos_points[-1]-(-2.279e+11))**2. + y_pos_points[-1]**2. + z_pos_points[-1]**2. )
            r_grid[i,j] = radius_error

        j += 1
    i += 1

# Make plots

# 2D Trajectory Plot
#plt.plot(x_pos_points,y_pos_points)
#plt.xlabel('x [meters]')
#plt.ylabel('y [meters]')
#plt.tick_params(labelright=True)
#plt.text(0,0, 'Sun',fontsize=8)
#plt.plot([2], [1], 'o')
#plt.axis('scaled')      # forces plot to be square
#plt.title('Simulated Earth Orbit - 1 Year')
#plt.grid(linestyle=':', linewidth='0.8')
'''
# 1D pos vs. time plots
plt.plot(tpoints,x_pos_points,label='X Position')
plt.plot(tpoints,y_pos_points,label='Y Position')
plt.plot(tpoints,z_pos_points,label='Z Position')
plt.title('Position vs. Time')
plt.legend()
'''




'''
#3D Plot
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x_pos_points, y_pos_points, z_pos_points, label='Orbit Trajectory')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
'''

# Mars-Sun L_1 is about 300 mars "atmospheric radii" (mars rad + 255 km)
#plt.imshow(r_grid, vmax=8.*au_distance,vmin=-0., origin='lower', aspect='equal', extent=[v_x_i_array[0],v_x_i_array[-1],v_y_i_array[0],v_y_i_array[-1]])
plt.imshow(r_grid, vmax=500.*3645000,vmin=3645000., cmap=cm.inferno_r, aspect='equal', extent=[v_x_i_array[0],v_x_i_array[-1],v_y_i_array[0],v_y_i_array[-1]])
plt.xlabel('Initial x velocity [m/s]')
plt.ylabel('Initial y velocity [m/s]')
plt.title('Distance From Mars at Simulation End')

#plt.colorbar()
cbar = plt.colorbar(ticks=[3645000., 100.*3645000., 200.*3645000., 300.*3645000., 400.*3645000., 500.*3645000.])
cbar.ax.set_yticklabels(['1 $r_{Mars \ atm.}$', '100 $r_{Mars \ atm.}$', '200 $r_{Mars \ atm.}$', '300 $r_{Mars \ atm.}$', '400 $r_{Mars \ atm.}$', '500 $r_{Mars \ atm.}$'])

plt.savefig('optimization_plot_mars_rad.pdf')

plt.clf()
#plt.show()

'''
plt.imshow(r_grid, cmap=cm.inferno_r, vmax=8.*au_distance,vmin=-0., origin='lower', aspect='equal', extent=[v_x_i_array[0],v_x_i_array[-1],v_y_i_array[0],v_y_i_array[-1]])
plt.xlabel('Initial x velocity [m/s]')
plt.ylabel('Initial y velocity [m/s]')
plt.title('Distance From Mars at Simulation End')


cbar = plt.colorbar(ticks=[0., 2.*au_distance, 4.*au_distance, 6.*au_distance, 8.*au_distance])
cbar.ax.set_yticklabels(['0 AU', '2 AU', '4 AU', '6 AU', '8 AU'])

plt.savefig('optimization_plot2.pdf')
'''
