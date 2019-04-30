"""HDBSCAN Clustering Tool
Author - Philip Ciunkiewicz
"""

###############################################################################
################################### Imports ###################################
###############################################################################


import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from clustering import *

import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog, messagebox, simpledialog
from ttkthemes import ThemedTk
from ttkwidgets import *


###############################################################################
################################## Globals ####################################
###############################################################################

# Setting the color theme to Seaborn defaults
sns.set()

# Creating a global figure and axes object for plotting
f = Figure(figsize=(10,10), dpi=80)
axes = [f.add_subplot(221), f.add_subplot(222), f.add_subplot(212)]
f.subplots_adjust(wspace=0.25, top=0.95, right=0.95, bottom=0.08)


###############################################################################
################################## Main GUI ###################################
###############################################################################


class mainGUI(Frame):
    """Main class for the entire GUI.
    """
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent

        self.init_data()
        self.init_controls_UI()
        self.init_plot_UI()

        self.ani = animation.FuncAnimation(f, self.animate, interval=100)


###############################################################################
################################## Plotting ###################################
###############################################################################


    def init_data(self):
        path = filedialog.askopenfilename(title='Select Data',
                                          initialdir=os.getcwd(), 
                                          filetypes=[("ThunderSTORM", "*.csv")])
        if path:
            self.df = pd.read_csv(path)
            self.XY = self.df[['x [nm]', 'y [nm]']]

            self.trim_sample()

            self.hdb = hdbscan.HDBSCAN(core_dist_n_jobs=6,
                                       gen_min_span_tree=True,
                                       min_cluster_size=1090,
                                       min_samples=629)
            self.hdb.fit(self.XY)
            plot_clusters_lite(self.XY, self.hdb.labels_, ax=axes[2])

    def trim_sample(self, n=200000):
        if self.XY.shape[0] > n:
                self.XY = self.XY.sample(n, random_state=1).values
        else:
            self.XY = self.XY.values

    def resize_data(self, xlim, ylim):
        mask = df['x [nm]'].between(*xlim) & df['y [nm]'].between(*ylim)
        self.XY = self.df[['x [nm]', 'y [nm]']][mask]
        self.trim_sample()
        
    def perform_clustering(self):
        self.hdb = hdbscan.HDBSCAN(core_dist_n_jobs=6,
                                   gen_min_span_tree=True,
                                   min_cluster_size=self.min_cluster_size.value,
                                   min_samples=self.min_samples.value)
        self.hdb.fit(self.XY)
        self.cluster.configure({'to': len(set(self.hdb.labels_)) - 2})
        axes[2].clear()
        self.animate(None, force_update=True)
        plot_clusters_lite(self.XY, self.hdb.labels_, ax=axes[2])
        self.cluster_info.config(text=full_cluster_info(self.hdb))

    def animate(self, i, force_update=False):
        """Main animation function for real-time plotting.
        """
        if self.changed_params() or force_update:
            axes[0].clear()
            axes[1].clear()
            view_cluster(self.hdb, self.XY, self.cluster.value,
                         p=self.probability.value/100, axes=axes)
            self.draw_region()
            self.draw_probability_marker()

    def changed_params(self):
        flag = False
        for key,item in self.current_values.items():
            if self.stored_values[key] != item.value:
                self.stored_values[key] = item.value
                flag=True
        
        return flag
    
    def draw_region(self):
        ymin, ymax = axes[1].get_ylim()
        xmin, xmax = axes[1].get_xlim()
        xspan = xmax - xmin
        yspan = ymax - ymin
        
        try:
            self.region.remove()
        except:
            pass
        
        self.region = Rectangle((xmin, ymin), xspan, yspan, 
                                linewidth=1.0, edgecolor=[1,0,0,1], 
                                facecolor='none', zorder=3)
        axes[2].add_patch(self.region)
        
    def draw_probability_marker(self):
        marker = axes[0].plot(self.probability.value/100, 0, 'r^', zorder=20)[0]
        marker.set_clip_on(False)


###############################################################################
################################# UI Elements #################################
###############################################################################


    def init_plot_UI(self):
        """Initializes the plot section of the GUI.
        """
        leftframe = Frame(root)
        leftframe.pack(side=LEFT,fill=BOTH, expand=True)
        
        container = Frame(leftframe,width=500, height=500)
        container.pack(side=TOP, fill=BOTH, expand=True)
        plot = plotFrame(container)
        plot.pack(fill=BOTH, expand=True)

    def init_controls_UI(self):
        """Initializes the controls section of the GUI.
        """
        self.rightframe = Frame(root)
        self.rightframe.pack(side=RIGHT,fill=BOTH, expand=True)

        header = Frame(self.rightframe)
        header.pack(side=TOP,fill=X)
        Label(header,text="Options",anchor='n',
              font=("Courier",16)).pack(side=LEFT,anchor=W)

        Separator(self.rightframe,orient=HORIZONTAL).pack(fill=X,pady=5)
        
        options = Frame(self.rightframe)
        cluster_n = Frame(options)
        Label(cluster_n, text="Cluster Index").pack(side=TOP, anchor=W)
        self.cluster = ScaleEntry(cluster_n, from_=0,
                                  to=len(set(self.hdb.labels_)) - 2,
                                  scalewidth=200,compound=LEFT)
        self.cluster.pack(side=BOTTOM)
        cluster_n.pack(anchor=W, side=TOP,fill=X)
        
        prob = Frame(options)
        Label(prob, text="Probability Threshold (%)").pack(side=TOP, anchor=W)
        self.probability = ScaleEntry(prob, from_=0, to=99, 
                                 scalewidth=200,compound=LEFT)
        self.probability.pack(side=BOTTOM)
        prob.pack(anchor=W, side=TOP,fill=X)
        options.pack(anchor=W, side=TOP,fill=X)

        Separator(self.rightframe,orient=HORIZONTAL).pack(fill=X,pady=5)

        minclustersize = Frame(self.rightframe)
        Label(minclustersize, text="Minimum Cluster Size").pack(side=TOP, anchor=W)
        self.min_cluster_size = ScaleEntry(minclustersize, from_=1, to=5000, 
                                 scalewidth=200,compound=LEFT)
        self.min_cluster_size.pack(side=BOTTOM)
        self.min_cluster_size._variable.set(1090)
        self.min_cluster_size._on_scale(None)
        minclustersize.pack(anchor=W, side=TOP,fill=X)
        
        minsamples = Frame(self.rightframe)
        Label(minsamples, text="Minimum Samples").pack(side=TOP, anchor=W)
        self.min_samples = ScaleEntry(minsamples, from_=1, to=1000, 
                                 scalewidth=200, compound=LEFT)
        self.min_samples.pack(side=BOTTOM)
        self.min_samples._variable.set(629)
        self.min_samples._on_scale(None)
        minsamples.pack(anchor=W, side=TOP, fill=X)
        
        controls = Frame(self.rightframe)
        self.run_hdbscan = Button(controls, text="Run HDBSCAN", 
                                command=self.perform_clustering)
        self.run_hdbscan.pack()
        controls.pack()
        
        Separator(self.rightframe, orient=HORIZONTAL).pack(fill=X,pady=5)
        
        self.cluster_info = Label(self.rightframe, text=full_cluster_info(self.hdb))
        self.cluster_info.pack(fill=X)
        
        
        self.current_values = {
        'cluster' : self.cluster,
        'probability' : self.probability,
        }
        self.stored_values = dict(self.current_values)


###############################################################################
################################### Classes ###################################
###############################################################################


class plotFrame(Frame):
    """Class for making the frame for the plot
    to be animated within, including TKinter
    drawing canvas and matplotlib toolbar.
    """
    def __init__(self, parent):
        Frame.__init__(self, parent)

        canvas = FigureCanvasTkAgg(f, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=BOTTOM, fill=BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)


###############################################################################
############################# Application Launch ##############################
###############################################################################


def clearterminal():
    """Clear the terminal window if using
    command line to launch the program.
    """
    if sys.platform.lower() == 'linux':
        os.system('clear')
    if sys.platform.lower() == 'win32':
        os.system('cls')


# Main function and class calls
if __name__ == "__main__":
    theme = "arc"
    themefg = "#5c616c"
    themebg = "#f5f6f7"

    clearterminal()
    root = ThemedTk(theme=theme)
    root.wm_title("HDBSCAN Clustering Utility v0.1")
    root.resizable(0,0)

    app = mainGUI(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
