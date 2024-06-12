from dedalus.extras import plot_tools
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import matplotlib

#Plot settings
dpi    = 144
ext    = '.h5'
outdir = 'plots_vels'
start  = 0
num    = 0
first  = 0
last   = 10000
if not os.path.exists(outdir):
    os.mkdir(outdir)

#Plot all data
all_files = os.listdir("snapshots/")
for files in all_files:
    if (files[-3:] == ext):
        filenumber = files[files.find("_s")+2:files.find(".")]
        if int(filenumber)%1 == 0:
            print("Plotting " + files)
            
            #Read data
            num = num + 1
            existing = outdir + '/' + 'plot{:04d}'.format(int(filenumber)+start) + '.png'
            print(existing)
            if not os.path.isfile(existing) and int(filenumber) >= first and int(filenumber) < last:
                file  = h5py.File("snapshots/"+files,"r")
                data = file['tasks']['bd']
                logbd = data[0].T

                data = file['tasks']['u']
                u     = data[0].T

                data = file['tasks']['v']
                v     = data[0].T

                data = file['tasks']['w']
                w     = data[0].T
                s        = data[0].T
                x    = data.dims[1][0]
                y    = data.dims[2][0]
                time = data.dims[0]['sim_time']
                
                #Plot data	
                cmap    = plt.get_cmap('magma') 
                fig     = plt.figure(figsize=(30,24))
                #fig.suptitle("t = {:4.3f}$T_0$".format(0.05*0.5*time/np.pi), fontsize=24)
                
                ax1 = fig.add_subplot(222)
                ax1.pcolormesh(x[:],y[:],s[:,:], cmap='magma')
                im = ax1.pcolormesh(x[:],y[:],s[:,:], cmap='magma')
                cb = plt.colorbar(im,shrink=0.75,format='%.0e')
                cb.ax.tick_params(labelsize=20)
                cb.set_label('Dust to gas ratio', rotation=270, labelpad=100, fontsize=50)
                ax1.set_xticks([])
                ax1.set_yticks([])
                
                ax2 = fig.add_subplot(224)
                ax2.pcolormesh(x[:],y[:],u[:,:], cmap=cmap)
                im = ax2.pcolormesh(x[:],y[:],u[:,:], cmap=cmap)
                cb = plt.colorbar(im,shrink=0.75,format='%.0e')
                cb.ax.tick_params(labelsize=20)
                cb.set_label('$u$', rotation=0,labelpad=40, fontsize=40)
                ax2.set_xticks([])
                ax2.set_yticks([])
                
                ax3 = fig.add_subplot(221)
                ax3.pcolormesh(x[:],y[:],v[:,:], cmap=cmap)
                im = ax3.pcolormesh(x[:],y[:],v[:,:], cmap=cmap)
                cb = plt.colorbar(im,shrink=0.75,format='%.0e')
                cb.ax.tick_params(labelsize=20)
                cb.set_label('$v$', rotation=0,labelpad=40, fontsize=40)
                ax3.set_xticks([])
                ax3.set_yticks([])
                
                ax4 = fig.add_subplot(223)
                ax4.pcolormesh(x[:],y[:],w[:,:], cmap=cmap)
                im = ax4.pcolormesh(x[:],y[:],w[:,:], cmap=cmap)
                cb = plt.colorbar(im,shrink=0.75,format='%.0e')
                cb.ax.tick_params(labelsize=20)
                cb.set_label('$w$', rotation=0,labelpad=40, fontsize=40)
                ax4.set_xticks([])
                ax4.set_yticks([])
                
                #Save the plot
                #fig.tight_layout()
                #plt.subplots_adjust(left=-0.5, bottom=0, right=1, top=1, wspace=-0.6, hspace=-0.45)
                #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                fig.savefig(outdir + '/' + 'plot{:04d}'.format(int(filenumber)+start), dpi=dpi)
                plt.close(fig)
