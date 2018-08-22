# Winn Koster  --  Final Project  --  Orbital DEQ proof of concept
# ephemeris: https://ssd.jpl.nasa.gov/horizons.cgi

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize
from scipy.optimize import fmin_tnc

G = 6.67408e-11
M = 1.989e+30
u = G*M
au_distance = 1.496e+11
#earth_distance = 1.49598023e+11
v_i = np.sqrt( u / au_distance )        # v initial is the earth

v_earth_i = np.array( [0.,v_i,0.] )
v_mars_f = np.array( [0.,-24162.,0.] )

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
#N = 1000000                             # setting this higher than 1000000 does NOT help the one year simulation at all. It's initial conditions at that point...
h = (t_1-t_0)/N

print(' ')
print(str(N)+' points in a simulation that lasts for '+str(t_1/31536000.)+' years.')
print('Points sampled using RK4 every '+str(h)+' seconds, or '+str(h/3600.)+' hours.')
print(' ')

# Takes initial velocity conditions array as ONE argument (v_i[0] = v_i_x, etc.), returns the distance from Mars using the given initial conditions. Will feed this to optimization routine to find where this function returns zero (or close to it)
def trajectory_simulation_error(v_i):
    # Begin numerical simulation...
    tpoints = np.arange(t_0,t_1,h)
    x_pos_points = []                   # x[0], the x position
    x_vel_points = []                   # x[1], the x velocity
    y_pos_points = []                   # y[0], the y position
    y_vel_points = []                   # y[1], the y velocity
    z_pos_points = []                   # z[0], the z position
    z_vel_points = []                   # z[1], the z velocity

    # initial conditions
    x_vec = np.array([au_distance, v_i[0] ],float)         # initial [x_pos,x_vel], will be updated upon iteration
    y_vec = np.array([0., v_i[1] ],float)         # initial [y_pos,y_vel], will be updated upon iteration
    z_vec = np.array([0., v_i[2]],float)         # initial [z_pos,z_vel], will be (eventually) updated upon iteration (when we add z variance)
    #z_vec = np.array([0., v_z_i],float)         # uncomment when you're ready to do 3d optimization. Add the v_z_i argument to the function too!


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
    print('Distance from Mars target: '+str(radius_error/1000.)+' km...')
    return radius_error

def trajectory_simulation_final_velocity(v_i):
    # Begin numerical simulation...
    tpoints = np.arange(t_0,t_1,h)
    x_pos_points = []                   # x[0], the x position
    x_vel_points = []                   # x[1], the x velocity
    y_pos_points = []                   # y[0], the y position
    y_vel_points = []                   # y[1], the y velocity
    z_pos_points = []                   # z[0], the z position
    z_vel_points = []                   # z[1], the z velocity

    # initial conditions
    x_vec = np.array([au_distance, v_i[0] ],float)         # initial [x_pos,x_vel], will be updated upon iteration
    y_vec = np.array([0., v_i[1] ],float)         # initial [y_pos,y_vel], will be updated upon iteration
    z_vec = np.array([0., v_i[2]],float)         # initial [z_pos,z_vel], will be (eventually) updated upon iteration (when we add z variance)
    #z_vec = np.array([0., v_z_i],float)         # uncomment when you're ready to do 3d optimization. Add the v_z_i argument to the function too!


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

    velocity = np.array( [x_vel_points[-1], y_vel_points[-1], z_vel_points[-1]] )
    return velocity


# initial guess for velocity vector
v_0_guess = np.array( [1.,32735.,1.] )

# example usage of minimize function
#res = minimize(trajectory_simulation_error, v_0_guess, method='nelder-mead', options={'maxiter': 20, 'disp': True})
#print(res.x)

stop_criterion = 3645000.       # the stop criterion. How close is considered close enough?
current_distance = stop_criterion + 10.     # just need something larger than stop_criterion to get the loop started; will update itself within the loop

while current_distance > stop_criterion:
    print('Looping through 50 more iterations of optimization algorithm...')
    print('Initial guess of '+str(v_0_guess))
    res = minimize(trajectory_simulation_error, v_0_guess, method='nelder-mead', options={'maxiter': 50, 'disp': True})
    print(res.x)
    print(' ')
    v_0_guess = res.x
    print('Updating v_0 guess to be '+str(res.x))
    current_distance = trajectory_simulation_error(res.x)
    print('Updating current distance to be '+str(current_distance))
    print(' ')


print('Satisfied with numerical convergence to within one mars atmospheric radius...')
print(str(current_distance/1000.)+' is less than 3645 km, this is good enough.')
print(' ')
print(res)

# Now let's define some orbital parameters, to see where we end up. Earth and Mars orbits are defined previously
# takes velocities as 3-vectors
# default orbits at earth and mars are LEO and Mars Recon. Orbiter, respectively
def orbital_params(v_spacecraft_i,v_spacecraft_f,v_earth_i,v_mars_f,v_orb_earth=7800.,v_orb_mars=3420.):
    v_i_relative = v_spacecraft_i - v_earth_i
    c_3_earth = np.sum( v_i_relative**2. )
    delta_v_earth = np.sqrt( c_3_earth + (np.sqrt(2)*v_orb_earth)**2. ) - v_orb_earth

    v_f_relative = v_spacecraft_f - v_mars_f
    c_3_mars = np.sum( v_f_relative**2. )
    delta_v_mars = np.sqrt( c_3_mars + (np.sqrt(2)*v_orb_mars)**2. ) - v_orb_mars

    burn_data = [ v_i_relative, c_3_earth, delta_v_earth, v_f_relative, c_3_mars, delta_v_mars ]
    print(' ')
    print('------------BURN DATA------------')
    print(' ')
    print('DEPARTURE CONDITIONS')
    print('Initial spacecraft velocity vector (relative to the Sun) [m/s]: '+str(v_spacecraft_i))
    print('Initial Earth velocity vector (relative to the Sun) [m/s]: '+str(v_earth_i))
    print('Initial spacecraft velocity vector (relative to the EARTH) [m/s]: '+str(v_i_relative))
    print('------------')
    print('Earth Departure C3 [m^2/s^2]: '+str(c_3_earth))
    print('Initial Earth Orbit Velocity [m/s]: '+str(v_orb_earth))
    print('Trans Martian Injection Burn delta-V [m/s]: '+str(delta_v_earth))
    print(' ')
    print('ARRIVAL CONDITIONS')
    print('Final spacecraft velocity vector (relative to the Sun) [m/s]: '+str(v_spacecraft_f))
    print('Final Mars velocity vector (relative to the Sun) [m/s]: '+str(v_mars_f))
    print('Final spacecraft velocity vector (relative to the MARS) [m/s]: '+str(v_f_relative))
    print('------------')
    print('Mars Arrival C3 [m^2/s^2]: '+str(c_3_mars))
    print('Target Mars Orbit Velocity [m/s]: '+str(v_orb_mars))
    print('Mars Orbital Insertion Burn delta-V [m/s]: '+str(delta_v_mars))
    print(' ')

    return burn_data


final_vel = trajectory_simulation_final_velocity(v_0_guess)

burn_data = orbital_params(v_0_guess, final_vel, v_earth_i, v_mars_f)


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

'''
plt.imshow(r_grid, vmax=8.*au_distance,vmin=-0., origin='lower', aspect='equal', extent=[v_x_i_array[0],v_x_i_array[-1],v_y_i_array[0],v_y_i_array[-1]])
plt.xlabel('Initial x velocity [m/s]')
plt.ylabel('Initial y velocity [m/s]')
plt.title('Distance From Mars at Simulation End')


cbar = plt.colorbar(ticks=[0., 2.*au_distance, 4.*au_distance, 6.*au_distance, 8.*au_distance])
cbar.ax.set_yticklabels(['0 AU', '2 AU', '4 AU', '6 AU', '8 AU'])

plt.savefig('optimization_plot1.pdf')

plt.clf()
#plt.show()


plt.imshow(r_grid, cmap=cm.inferno_r, vmax=8.*au_distance,vmin=-0., origin='lower', aspect='equal', extent=[v_x_i_array[0],v_x_i_array[-1],v_y_i_array[0],v_y_i_array[-1]])
plt.xlabel('Initial x velocity [m/s]')
plt.ylabel('Initial y velocity [m/s]')
plt.title('Distance From Mars at Simulation End')


cbar = plt.colorbar(ticks=[0., 2.*au_distance, 4.*au_distance, 6.*au_distance, 8.*au_distance])
cbar.ax.set_yticklabels(['0 AU', '2 AU', '4 AU', '6 AU', '8 AU'])

plt.savefig('optimization_plot2.pdf')
'''
