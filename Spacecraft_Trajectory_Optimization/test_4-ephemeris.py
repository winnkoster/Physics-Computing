# Winn Koster  --  Final Project  --  Trajectory optimization using ephemeris data for 2024 transfer window
# ephemeris: https://ssd.jpl.nasa.gov/horizons.cgi

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ------------------- LOAD THE EPHEMERIS DATA -------------------

# Load the earth data, which is from July 1, 2024 through December 31, 2024.
# Load the mars data, which is from November 28,2024 through February 21, 2026.
#           Data format is...
#           data[0] = [time[0], x_pos[0], y_pos[0], z_pos[0], x_vel[0], y_vel[0], z_vel[0] ]
# We skip over column 1 because it encodes the time in "A.D. 2026-Feb-20 00:00:00.0000" format, while column 0 encodes as "2461091.500000000"
# This way, stepping +1.0 is one day later, and is writable to a numpy array out of the box
# We will set July 1, 2024, 00:00 UTC as time=0 within the sanitization loop
# These numbers are in KILOMETERS FOR SECOND so you need to convert them before using them in your sim

earth_data = np.genfromtxt('earth_departure_data.txt', delimiter=',', skip_header=57, max_rows=184, usecols=(0,2,3,4,5,6,7) )
mars_data = np.genfromtxt('mars_arrival_data.txt', delimiter=',', skip_header=52, max_rows=451, usecols=(0,2,3,4,5,6,7) )


# Now begin the sanitization routine. This gets us distances and velocities in SI units, and sets July 1, 2024, 00:00 UTC as time=0.
i = 0
while i < len(earth_data):
    # We will set July 1, 2024, 00:00 UTC as time=0.
    earth_data[i,0] -= 2460492.5

    # Multiply distances by 1000 to switch from km to m
    earth_data[i,1] *= 1000.
    earth_data[i,2] *= 1000.
    earth_data[i,3] *= 1000.

    # Multiply velocities by 1000 to switch from km/s to m/s
    earth_data[i,4] *= 1000.
    earth_data[i,5] *= 1000.
    earth_data[i,6] *= 1000.

    i += 1

i = 0
while i < len(mars_data):
    # We will set July 1, 2024, 00:00 UTC as time=0.
    mars_data[i,0] -= 2460492.5

    # Multiply distances by 1000 to switch from km to m
    mars_data[i,1] *= 1000.
    mars_data[i,2] *= 1000.
    mars_data[i,3] *= 1000.

    # Multiply velocities by 1000 to switch from km/s to m/s
    mars_data[i,4] *= 1000.
    mars_data[i,5] *= 1000.
    mars_data[i,6] *= 1000.

    i += 1




# ------------------- DEFINE CONSTANTS AND FUNCTIONS -------------------

G = 6.67408e-11
M = 1.989e+30
u = G*M
au_distance = 1.496e+11


# in the DEQ below, we have made the following substitution to turn a 2nd order diffeq into 2 coupled first order diffeqs...
# OG equation in radial...
# decomposed into cartesian...              \ddot x \ = \ \frac{-\mu x}{(x^2 + y^2 + z^2)^{3/2}}
# then we set x_dot as the substitution dummy var

def x_pos_2d(x_vec,y_vec,z_vec,t):                  # x_vec[] is an array that encodes the x-location x[0] and the x-velocity x[1]. We've called this 'v_x', so that we may maintain using 'd_' as a prefix (eg: d_x and d_v_x) in order to be consistent with other codes
    x = x_vec[0]
    y = y_vec[0]
    z = z_vec[0]
    v_x = x_vec[1]
    d_x = v_x
    d_v_x = -(u*x)/(( (x**2.) + (y**2.) + (z**2.) )**(3./2.) )
    return np.array([d_x,d_v_x],float)

def y_pos_2d(x_vec,y_vec,z_vec,t):                  # x_vec[] is an array that encodes the x-location x[0] and the x-velocity x[1]. We've called this 'v_x', so that we may maintain using 'd_' as a prefix (eg: d_x and d_v_x) in order to be consistent with other codes
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

# Takes v_i as a 3-vector
def trajectory_simulation_error(v_i):               # Takes initial velocity conditions array as ONE argument (v_i[0] = v_i_x, etc.), returns the distance from Mars using the given initial conditions. Will feed this to optimization routine to find where this function returns zero (or close to it)
    # Begin numerical simulation...
    t_0 = earth_params[0]*1440.*60.             # converts argument given in days into seconds by multiplying by 1440 minutes in a day, then 60 seconds in a minute
    t_1 = mars_params[0]*1440.*60.
    N = 8000                              # for fast test runs, don't use for real sims.
    #N = 40000                               # medium length. This seems to be ideal for grad samples. IF we're only doing one trajectory, consider using the one below
    #N = 1000000                            # setting this higher than 1000000 does NOT help the one year simulation at all. It's initial conditions at that point...
    h = (t_1-t_0)/N


    tpoints = np.arange(t_0,t_1,h)
    x_pos_points = []                   # x[0], the x position
    x_vel_points = []                   # x[1], the x velocity
    y_pos_points = []                   # y[0], the y position
    y_vel_points = []                   # y[1], the y velocity
    z_pos_points = []                   # z[0], the z position
    z_vel_points = []                   # z[1], the z velocity

    # initial conditions. We're starting at earth, so that's our initial location. Initial velocity guess is taken as function argument
    x_vec = np.array([earth_params[1], v_i[0] ],float)         # initial [x_pos,x_vel], will be updated upon iteration
    y_vec = np.array([earth_params[2], v_i[1] ],float)         # initial [y_pos,y_vel], will be updated upon iteration
    z_vec = np.array([earth_params[3], v_i[2]],float)         # initial [z_pos,z_vel], will be updated upon iteration

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
    radius_error = np.sqrt( (x_pos_points[-1]-mars_params[1])**2. + (y_pos_points[-1]-mars_params[2])**2. + (z_pos_points[-1]-mars_params[2])**2. )
    print('Distance from Mars target: '+str(radius_error/1000.)+' km...')
    return radius_error

def trajectory_simulation_final_velocity(v_i):      # returns the final velocity as a vector given initial conditions. Will be run to determine c_3 and dV stats once optimization algorithm has converged
    # Begin numerical simulation...
    t_0 = earth_params[0]*1440.*60.             # converts argument given in days into seconds by multiplying by 1440 minutes in a day, then 60 seconds in a minute
    t_1 = mars_params[0]*1440.*60.
    #N = 10000                              # for fast test runs, don't use for real sims.
    N = 40000                               # medium length. This seems to be ideal for grad samples. IF we're only doing one trajectory, consider using the one below
    #N = 1000000                            # setting this higher than 1000000 does NOT help the one year simulation at all. It's initial conditions at that point...
    h = (t_1-t_0)/N


    tpoints = np.arange(t_0,t_1,h)
    x_pos_points = []                   # x[0], the x position
    x_vel_points = []                   # x[1], the x velocity
    y_pos_points = []                   # y[0], the y position
    y_vel_points = []                   # y[1], the y velocity
    z_pos_points = []                   # z[0], the z position
    z_vel_points = []                   # z[1], the z velocity

    # initial conditions. We're starting at earth, so that's our initial location. Initial velocity guess is taken as function argument
    x_vec = np.array([earth_params[1], v_i[0] ],float)         # initial [x_pos,x_vel], will be updated upon iteration
    y_vec = np.array([earth_params[2], v_i[1] ],float)         # initial [y_pos,y_vel], will be updated upon iteration
    z_vec = np.array([earth_params[3], v_i[2]],float)         # initial [z_pos,z_vel], will be updated upon iteration

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
    print('Mars Arrival C3 [m^2/s^2]: '+str(c_3_earth))
    print('Target Mars Orbit Velocity [m/s]: '+str(v_orb_mars))
    print('Mars Orbital Insertion Burn delta-V [m/s]: '+str(delta_v_mars))
    print(' ')

    return burn_data

def optimization_algorithm(v_0_guess,earth_params_arg,mars_params_arg):

    # explicitly defining the arrays within the function, so that the optimize function still takes only one argument.
    # As much as I'd love to have an orbit funciton that takes three unique arguments, scipy NEEDS a one argument function to optimize.
    # So we simply accept the earth and mars data as arguments for THIS function, and then define the variables explicitly for the "sub functions" to use

    # we set the variables called within this function to be global, and since this function calls the trajectory function, we effectively set the arguments without actually needing to provide them as arguments
    global earth_params
    global mars_params

    earth_params = earth_params_arg
    mars_params = mars_params_arg

    print('Running optimization algorithm...')
    print(' ')
    print('-------------------- Earth and Mars departure and arrival conditions --------------------')
    print('Earth departure date is '+str(earth_params[0])+' days after July 1, 2024.')
    print('Earth x position vector [m]: '+str(earth_params[1]))
    print('Earth y position vector [m]: '+str(earth_params[2]))
    print('Earth z position vector [m]: '+str(earth_params[3]))
    print('Earth x velocity vector [m/s]: '+str(earth_params[4]))
    print('Earth y velocity vector [m/s]: '+str(earth_params[5]))
    print('Earth z velocity vector [m/s]: '+str(earth_params[6]))
    print(' ')

    print('Mars arrival date is '+str(mars_params[0])+' days after July 1, 2024.')
    print('Mars x position vector [m]: '+str(mars_params[1]))
    print('Mars y position vector [m]: '+str(mars_params[2]))
    print('Mars z position vector [m]: '+str(mars_params[3]))
    print('Mars x velocity vector [m/s]: '+str(mars_params[4]))
    print('Mars y velocity vector [m/s]: '+str(mars_params[5]))
    print('Mars z velocity vector [m/s]: '+str(mars_params[6]))
    print(' ')

    #stop_criterion = 3645000.       # the stop criterion. How close is considered close enough?
    stop_criterion = 100000000.      # The dV required between here and anywhere closer is going to be 5ish m/s or less, this is fine
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
    velocity_vector = res.x

    print('Satisfied with numerical convergence to within 100,000 km of the Mars center of mass...')
    print(str(current_distance/1000.)+' is less than 100,000 km, this is good enough.')
    print(' ')
    print(res)

    return velocity_vector




# -------------------- Test --------------------
test_run1 = False
test_run2 = False
test_run3 = True

# TEST 1 - simulates a trajectory, returns the distance from mars target
if test_run1 == True:
    print('Test variable 1 is true, running test case for TRAJECTORY SIMULATION...')
    print(' ')
    print('Earth Params [timestamp, 3-vec position, 3-vec velocity]:')
    earth_params = earth_data[10]
    print(earth_params)
    print('Mars Params [timestamp, 3-vec position, 3-vec velocity]:')
    mars_params = mars_data[100]
    print(mars_params)

    print(' ')
    print('Initial Guess:')
    guess = np.array( [earth_params[4]*1.5,earth_params[5]*1.5,earth_params[6]] )
    print(guess)

    print(' ')
    print('Distance from target and final velocity:')
    print( trajectory_simulation_error(guess) )
    print( trajectory_simulation_final_velocity(guess) )
else:
    print('Test variable 1 is false, skipping trajectory simulation test...')

# TEST 2 - optimizes the trajectory, returns the velocity vector optimization
if test_run2 == True:
    print('Test variable 2 is true, running test case for OPTIMIZATION...')
    print(' ')
    print('Initial Guess:')
    guess = np.array( [17000., 23000. , -16500.] )        # a pretty good guess that should converge
    print(guess)

    print(' ')
    print('Optimized velocity vector:')
    print( optimization_algorithm(guess, earth_data[30], mars_data[150]) )
else:
    print('Test variable 2 is false, skipping trajectory optimization test...')

# TEST 3 - determines the delta-V (and other) stats for the optimized trajectory
if test_run3 == True:
    print('Test variable 3 is true, running test case for TRAJECTORY ANALYSIS...')
    print(' ')
    guess = np.array( [17000., 23000. , -16500.] )        # a pretty good guess that should converge
    print(' ')

    # building args for orbital_params() function...
    v_spacecraft_i = optimization_algorithm(guess, earth_data[30], mars_data[150])
    v_spacecraft_f = trajectory_simulation_final_velocity(v_spacecraft_i)
    v_earth_i = np.array( [ earth_data[30,4],earth_data[30,5],earth_data[30,6] ] )
    v_mars_f = np.array( [ mars_data[150,4],mars_data[150,5],mars_data[150,6] ] )

    aa = orbital_params(v_spacecraft_i,v_spacecraft_f,v_earth_i,v_mars_f)
    print(' ')
    print(aa)
else:
    print('Test variable 3 is false, skipping trajectory analysis test...')
