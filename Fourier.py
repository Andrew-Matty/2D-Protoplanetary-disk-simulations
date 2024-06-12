from dedalus.extras import plot_tools
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
from scipy.fft import fft
from scipy.interpolate import interp1d

# In[2]:
#Plot settings
dpi    = 144
ext    = '.h5'
outdir = 'Fplots'
start  = 0
number = 0
first  = 0
last   = 1000
if not os.path.exists(outdir):
        os.mkdir(outdir)

#Plot all data
all_files = os.listdir("snapshots/")
for files in all_files:
    if (files[-3:] == ext):
        filenumber = files[files.find("_s")+2:files.find(".")]
                
        #Read data
        number = number + 1
        existing = outdir + '/' + 'plot{:04d}'.format(int(filenumber)+start) + '.png'
        if not os.path.isfile(existing) and number%1 == 0 and int(filenumber) >= first and int(filenumber) < last:
            
            #Read the h5py file
            file = h5py.File("snapshots/"+files,"r")
            try:
                #x        = file['scales/x/1']
                #y        = file['scales/z/1']
                #vx       = file['tasks/ud']
                #vy       = file['tasks/wd']
                #s        = file['tasks/bd']
                #time = file['scales/sim_time'][0]

                data     = file['tasks']['bd']
                s        = data[0].T
                x        = data.dims[1][0]
                y        = data.dims[2][0]
                time     = data.dims[0]['sim_time']
            except:
                continue
            
            #Store the data into lists to be able to plot it            
            bds = []
            for i in range(1):
                for j in range(256):
                    bds.append(s[i,j])
                    
            # Number of sample points
            N = 512

            # sample spacing
            T = 1.0/N
            xf = np.linspace(0.0, 0.2*N*T, N)
            bds = np.array(bds)
            bds = bds - np.mean(bds)
            yf = fft(bds)
            
            #Plot data
            fig     = plt.figure(figsize=(12,6))
            fig.suptitle("t = {:4.3f}$T_0$".format(0.1*time[0]/(2.0*np.pi)), fontsize=15, x=0.45, y=0.95)                
            plt.xlabel('Frequency', fontsize=15)
            plt.ylabel('Apmplitude', fontsize=15)
            plt.xlim(0,20)
                
            xf = np.linspace(0.0, 1.0//(2.0*T), N//2)
            yf2 = 2.0/N*np.abs(yf[0:N//2])
            norm = np.max(yf2)
            yf2_norm = yf2/norm
            #plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))

            ft_interp = interp1d(xf,yf2_norm,kind='cubic')
            x_plot = np.linspace(0.0, 20, 10000)
            print("Plotting " + files)
            plt.plot(x_plot, ft_interp(x_plot))

            #Save the plot
            #plt.axis('off')
            fig.savefig(outdir + '/' + 'plot{:04d}'.format(int(filenumber)+start), dpi=dpi)
            plt.close(fig)
