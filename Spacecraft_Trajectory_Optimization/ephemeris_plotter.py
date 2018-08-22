import numpy as np
import matplotlib.pyplot as plt

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

mars_x_pts = []
mars_y_pts = []
earth_x_pts = []
earth_y_pts = []

i = 0
while i < len(mars_data):
    mars_x_pts.append( mars_data[i,1] )
    mars_y_pts.append( mars_data[i,2] )

    i += 1

i = 0
while i < len(earth_data):
    earth_x_pts.append( earth_data[i,1] )
    earth_y_pts.append( earth_data[i,2] )

    i += 1


# 2D Trajectory Plot
plt.plot(earth_x_pts,earth_y_pts)
plt.plot(mars_x_pts,mars_y_pts,color='Red')
plt.xlabel('x [meters]')
plt.ylabel('y [meters]')
plt.tick_params(labelright=True)


plt.text(earth_x_pts[0]-(1.0e+11),earth_y_pts[0]+(0.4e+11), 'Earth (initial positions)', fontsize=8)         # earth text
plt.text(0,0, 'Sun',fontsize=8)   # sun text
plt.text(mars_x_pts[0]+(0.1e+11),mars_y_pts[0], 'Mars (final positions)', fontsize=8)       # mars text
plt.plot([2], [1], 'o')         # sun point



plt.axis('scaled')      # forces plot to be square
plt.title('Earth Departure and Mars Arrival Locations')
plt.grid(linestyle=':', linewidth='0.8')

plt.savefig('ephemeris_positions.png',dpi=300)
#plt.show()
