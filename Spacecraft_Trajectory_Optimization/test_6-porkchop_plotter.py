import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------- Import the data ----------

# extent=[27.,187.,107.,247.]
#velocity_array_earth_only = np.loadtxt('velocity_earth_test2.txt')
#velocity_array_mars_only = np.loadtxt('velocity_mars_test2.txt')
#velocity_array_tot = np.loadtxt('velocity_tot_test2.txt')

velocity_array_earth_only = np.loadtxt('velocity_earth.txt')
velocity_array_mars_only = np.loadtxt('velocity_mars.txt')
velocity_array_tot = np.loadtxt('velocity_tot.txt')

# ------ sanitize / format -----
# np likes to read vertical as first entry and horizontal as second entry.
# this is the opposite to what english speakers and matplotlib does. We convert to something more sensical here
velocity_array_earth_only = np.transpose(velocity_array_earth_only)
velocity_array_mars_only = np.transpose(velocity_array_mars_only)
velocity_array_tot = np.transpose(velocity_array_tot)

# The previous code returns a super high number (~1.479420942155084922e+05 on the departure speed, varies on the arrival speed)
# In retrospect, I should have just saved them as NaNs, but whatever, we can do that here...

i=0
while i < len(velocity_array_earth_only):
    j=0
    while j < len(velocity_array_earth_only[0]):
        if velocity_array_earth_only[i,j] > 1.469420942155084922e+05:
            velocity_array_earth_only[i,j] = np.nan
            velocity_array_mars_only[i,j] = np.nan
            velocity_array_tot[i,j] = np.nan
        j +=1
    i += 1




# ---------- plots! ----------

plt.imshow(velocity_array_earth_only, vmax=30000., vmin=2000., cmap=cm.inferno_r, aspect='equal', extent=[0.,184.,107.,307.], origin="lower")
plt.xlabel('Departure Date')
plt.ylabel('Arrival Date')
plt.title('Earth Departure $\Delta v$')
plt.xticks((0,184),['7/1/2024','12/31/2024'])
plt.yticks((107,307),['3/15/2025','10/1/2025'])
plt.colorbar()
cm.inferno_r.set_bad(color='grey')
plt.savefig('./porkchop_plots/earth.pdf')
plt.clf()


plt.imshow(velocity_array_mars_only, vmax=30000., vmin=2000., cmap=cm.inferno_r, aspect='equal', extent=[0.,184.,107.,307.], origin="lower")
plt.xlabel('Departure Date')
plt.ylabel('Arrival Date')
plt.title('Mars Arrival $\Delta v$')
plt.xticks((0,184),['7/1/2024','12/31/2024'])
plt.yticks((107,307),['3/15/2025','10/1/2025'])
plt.colorbar()
cm.inferno_r.set_bad(color='grey')
plt.savefig('./porkchop_plots/mars.pdf')
plt.clf()

plt.imshow(velocity_array_tot, vmax=30000., vmin=5000., cmap=cm.inferno_r, aspect='equal', extent=[0.,184.,107.,307.], origin="lower")
plt.xlabel('Departure Date')
plt.ylabel('Arrival Date')
plt.title('Mars Injection and Insertion (total) $\Delta v$')
plt.xticks((0,184),['7/1/2024','12/31/2024'])
plt.yticks((107,307),['3/15/2025','10/1/2025'])
plt.colorbar()
cm.inferno_r.set_bad(color='grey')
plt.savefig('./porkchop_plots/total.pdf')
plt.clf()






#
