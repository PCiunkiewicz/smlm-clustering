"""
Author: Philip Ciunkiewicz
"""


###############################################################################
################################### Imports ###################################
###############################################################################


import matplotlib
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

import os
import sys
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from time import time
from scipy.stats import randint

from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog, messagebox, simpledialog
from ttkthemes import ThemedTk
from ttkwidgets import *
from ttkwidgets.frames import Balloon

import UI
from clustering import *


###############################################################################
################################## Settings ###################################
###############################################################################


matplotlib.use('TkAgg')
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.seterr(divide='ignore')
sns.set()


###############################################################################
################################## Main GUI ###################################
###############################################################################


class mainGUI(Frame):
    """Main class for the entire GUI.
    """
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.init_vars()

        self.plot = UI.plotFrame(root, f)
        self.plot.pack(side=LEFT, fill=BOTH, expand=True)

        self.controls = UI.controlsFrame(root, self)
        self.controls.pack(side=RIGHT, fill=BOTH, expand=False, padx=5, anchor=W)

        self.ani = animation.FuncAnimation(f, self.animate, interval=100)


    def init_vars(self):
        self.clustermin = IntVar()
        self.clustermin.set(2)
        self.clustermax = IntVar()
        self.clustermax.set(100)

        self.samplemin = IntVar()
        self.samplemin.set(1)
        self.samplemax = IntVar()
        self.samplemax.set(50)

        self.allowsingle = BooleanVar()
        self.silhouette = BooleanVar()
        self.calinski = BooleanVar()
        self.davies = BooleanVar()

        self.prev_ROI = [0, 0, 0, 0]


###############################################################################
################################ Data Handling ################################
###############################################################################


    def load_data(self):
        path = filedialog.askopenfilename(title='Select Data',
                                          initialdir=os.getcwd(), 
                                          filetypes=[("ThunderSTORM", "*.csv")])
        if path:
            self.filename = os.path.split(path)[1][:-4]
            self.df = pd.read_csv(path)
            self.XY = self.df[['x [nm]', 'y [nm]']]

            self.subsample_data()

            for ax in axes:
                ax.clear()

            self.min_cluster_size.configure({'to': min([self.XY.shape[0]//10, 5000])})
            self.min_samples.configure({'to': min([self.XY.shape[0]//10, 1000])})
            plot_clusters_lite(self.XY, -np.ones(self.XY.shape[0]), ax=axes[2])
            self.cluster_info.config(text='')

            self.ROI()


    def subsample_data(self, n=500000):
        if self.XY.shape[0] > n:
            warn = 'Would you like to subsample your dataset for better performance?'
            if messagebox.askyesno('Large Dataset Detected', warn):
                self.XY = self.XY.sample(n, random_state=1).values
            else:
                self.XY = self.XY.values
        else:
            self.XY = self.XY.values


    def ROI(self, update=False, reset=False):
        xlim, ylim = axes[2].get_xlim(), axes[2].get_ylim()
        new_ROI = [*xlim, *ylim]
        if self.prev_ROI != new_ROI:
            self.prev_ROI = new_ROI
            ROI_text = inspect.cleandoc(f"""
                                        X Range:   {xlim[0]:.0f}  --  {xlim[1]:.0f}
                                        Y Range:   {ylim[0]:.0f}  --  {ylim[1]:.0f}
                                        """)
            self.ROI_info.config(text=ROI_text)

        try:
            if update:
                mask = self.df['x [nm]'].between(*xlim) & self.df['y [nm]'].between(*ylim)

            if reset:
                mask = np.ones(self.df.shape[0], dtype=bool)

            if update or reset:
                self.XY = self.df[['x [nm]', 'y [nm]']][mask].values
                axes[2].clear()
                plot_clusters_lite(self.XY, -np.ones(self.XY.shape[0]), ax=axes[2])
                self.ROI()

        except:
            messagebox.showinfo('Reset ROI','Please load data first.')


###############################################################################
########################### Computational Functions ###########################
###############################################################################

        
    def perform_clustering(self):
        try:
            _ = self.XY
        except:
            messagebox.showinfo('Run HDBSCAN','Please load data first.')
            return

        start = time()
        self.hdb = hdbscan.HDBSCAN(core_dist_n_jobs=6,
                                   gen_min_span_tree=True,
                                   min_cluster_size=self.min_cluster_size.value,
                                   min_samples=self.min_samples.value,
                                   allow_single_cluster=self.allowsingle.get())
        self.hdb.fit(self.XY)
        self.cluster.configure({'to': len(set(self.hdb.labels_[self.hdb.labels_ != -1])) - 1})
        axes[2].clear()
        self.animate(None, force_update=True)
        plot_clusters_lite(self.XY, self.hdb.labels_, ax=axes[2])
        self.cluster_info.config(text=full_cluster_info(self.hdb))
        messagebox.showinfo(f'HDBSCAN Results ({time() - start:.1f} seconds)', 
                            full_cluster_info(self.hdb))


    def optimize_params(self):
        try:
            _ = self.XY
        except:
            messagebox.showinfo('Run HDBSCAN','Please load data first.')
            return

        n = simpledialog.askinteger('Parameter Search', 'Enter number of searches:')
        if n:
            param_dist = {"min_cluster_size": randint(self.clustermin.get(), 
                                                      self.clustermax.get()),
                          "min_samples": randint(self.samplemin.get(), 
                                                 self.samplemax.get())}

            self.results = random_search_custom_hdb(param_dist, self.XY, n=n,
                                                    allowsingle=self.allowsingle.get(),
                                                    silhouette=self.silhouette.get(),
                                                    calinski=self.calinski.get(),
                                                    davies=self.davies.get())
            self.results.to_csv(f'Params/Parameter_search_{self.filename}.csv', index=False)
            self.plot_param_search()


###############################################################################
############################# Plotting Functions ##############################
###############################################################################


    def animate(self, i, force_update=False):
        """Main animation function for real-time plotting.
        """
        if self.changed_params() or force_update:
            try:
                axes[0].clear()
                axes[1].clear()
                view_cluster(self.hdb, self.XY, self.cluster.value,
                             p=self.probability.value/100, axes=axes)
                self.draw_region()
                self.draw_probability_marker()
            except:
                pass


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


    def plot_param_search(self):
        try:
            self.results_window.destroy()
        except:
            pass

        self.results_window = Toplevel(root, background=themebg)
        self.param_fig = Figure(figsize=(8,6), dpi=80, facecolor=themebg)
        self.param_ax = self.param_fig.add_subplot(111)
        self.param_fig.subplots_adjust(right=1.0)

        self.results_plot = UI.plotFrame(self.results_window, self.param_fig)
        self.results_plot.pack(side=LEFT, fill=BOTH, expand=True)
        heatmap = self.results.pivot('min_samples', 'min_cluster_size', 'score')
        sns.heatmap(heatmap, ax=self.param_ax, cmap='RdYlGn')

        var = StringVar()
        scores = self.results.drop(['min_samples', 'min_cluster_size'], 
                                   axis=1).columns.values
        OptionMenu(self.results_window, var, 'Select Plot', *scores, 
                   command=self.update_param_plot).pack(side=BOTTOM)

        self.draw_param_table()
        self.results_window.geometry("1000x600+0+0")


    def update_param_plot(self, score):
        self.param_fig.clear(keep_observers=True)
        self.param_ax = self.param_fig.add_subplot(111)
        heatmap = self.results.pivot('min_samples', 'min_cluster_size', score)
        sns.heatmap(heatmap, ax=self.param_ax, cmap='RdYlGn')
        self.results_plot.canvas.draw()


    def draw_param_table(self):
        subframe = Frame(self.results_window)
        table = UI.displayTable(self.results_window, self.results)
        subframe.pack(side=LEFT, padx=5)

        self.make_header(subframe, "Parameter Search Results")


###############################################################################
############################### Helper Functions ##############################
###############################################################################


    def changed_params(self):
        flag = False
        for key,item in self.current_values.items():
            if self.stored_values[key] != item.value:
                self.stored_values[key] = item.value
                flag=True
        
        return flag

    
    def make_header(self, frame, headertext, target=None, state=False):
        header = Frame(frame)
        header.pack(side=TOP, fill=X)
        Label(header, text=headertext,anchor='n',
              font=("Courier",16)).pack(side=LEFT, anchor=W, pady=(25,0))
        Separator(frame, orient=HORIZONTAL).pack(fill=X,pady=(5,1))
        Separator(frame, orient=HORIZONTAL).pack(fill=X,pady=(1,5))

        if target is not None:
            UI.frameCollapser(header, target, state)


    def make_select(self, frame, text, var):
        subframe = Frame(frame)
        Checkbutton(subframe, variable=var, text=text).pack(side=BOTTOM, anchor=W)
        subframe.pack(anchor=W, side=TOP, fill=X)


def clearterminal():
    """Clear the terminal window if using
    command line to launch the program.
    """
    if sys.platform.lower() == 'linux':
        os.system('clear')
    if sys.platform.lower() == 'win32':
        os.system('cls')


###############################################################################
############################# Application Launch ##############################
###############################################################################


# Main function and class calls
if __name__ == "__main__":
    theme = "arc"
    themefg = "#5c616c"
    themebg = "#f5f6f7"

    # Creating a global figure and axes object for plotting
    f = Figure(figsize=(10,10), dpi=80, facecolor=themebg)
    axes = [f.add_subplot(221), f.add_subplot(222), f.add_subplot(212)]
    f.subplots_adjust(wspace=0.25, top=0.95, right=0.95, bottom=0.08)

    clearterminal()
    root = ThemedTk(theme=theme)
    root.wm_title("HDBSCAN Clustering Utility v0.3.1")
    root.configure(background=themebg)

    app = mainGUI(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
