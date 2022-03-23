"""
Author: Philip Ciunkiewicz
"""
import os
import sys
import inspect
import warnings
import multiprocessing
from time import time

import hdbscan
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import tkinter as tk
import matplotlib.animation as animation
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.stats import randint
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from tkinter import ttk, filedialog, messagebox, simpledialog
from ttkthemes import ThemedTk

import UI
from clustering.plot import plot_clusters, view_cluster, view_silhouette
from clustering.search import random_search
from clustering.stats import full_cluster_info


###############################################################################
################################## Settings ###################################
###############################################################################


if getattr( sys, 'frozen', False ) :
    os.chdir(os.path.dirname(os.getcwd()))
    multiprocessing.freeze_support()

matplotlib.use('TkAgg')
warnings.filterwarnings("ignore", category=DeprecationWarning)
np.seterr(divide='ignore')
sns.set()


###############################################################################
################################## Main GUI ###################################
###############################################################################


class mainGUI(ttk.Frame):
    """Main class for the entire GUI.
    """
    def __init__(self, parent):
        super().__init__(parent)
        self.init_vars()

        self.plot = UI.PlotFrame(root, f)
        self.plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.controls = UI.ControlsFrame(root, self)
        self.controls.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False, padx=5, anchor=tk.W)

        self.ani = animation.FuncAnimation(f, self.animate, interval=100)


    def init_vars(self):
        self.clustermin = tk.IntVar()
        self.clustermin.set(2)
        self.clustermax = tk.IntVar()
        self.clustermax.set(100)

        self.samplemin = tk.IntVar()
        self.samplemin.set(1)
        self.samplemax = tk.IntVar()
        self.samplemax.set(50)

        self.allowsingle = tk.BooleanVar()
        self.hidenotice = tk.BooleanVar()
        self.silhouette = tk.BooleanVar()
        self.calinski = tk.BooleanVar()
        self.davies = tk.BooleanVar()

        self.clust_method = tk.StringVar()
        self.clust_method.set('eom')
        self.validation_mode = tk.StringVar()
        self.validation_mode.set('Relative Validity')

        self.prev_ROI = [0, 0, 0, 0]


###############################################################################
################################ Data Handling ################################
###############################################################################


    def load_data(self):
        """Prompts user for path to data and loads
        the data into a Pandas DataFrame. Updates
        certain GUI sliders based on the size of the
        dataset.
        """
        path = filedialog.askopenfilename(title='Select Data',
                                          initialdir=os.getcwd(),
                                          filetypes=[("ThunderSTORM", "*.csv")])
        if path:
            self.filename = os.path.split(path)[1][:-4]
            self.df = pd.read_csv(path)
            try:
                self.XY = self.df[['x [nm]', 'y [nm]']]
            except KeyError:
                success = self.convert_columns()
                if success:
                    self.XY = self.df[['x [nm]', 'y [nm]']]
                else:
                    return

            self.subsample_data()

            for ax in axes:
                ax.clear()

            self.min_cluster_size.configure({'to': min([self.XY.shape[0]//10, 5000])})
            self.min_samples.configure({'to': min([self.XY.shape[0]//10, 1000])})
            plot_clusters(self.XY, -np.ones(self.XY.shape[0]), ax=axes[2])
            self.cluster_info.config(text='')

            self.plot.canvas.draw()
            self.ROI()


    def convert_columns(self):
        """Rename columns if they do not follow
        ThunderSTORM convention.
        """
        prompt = inspect.cleandoc(f"""Unable to automatically identify columns.
                                  Please provide the column index / letter for
                                  each of the following options.""")
        messagebox.showinfo('Load Data', prompt)
        xcol = simpledialog.askstring('Load Data', 'Identify X column.')
        ycol = simpledialog.askstring('Load Data', 'Identify Y column.')
        # intensitycol = simpledialog.askstring('Load Data', 'Identify intensity column.')

        for item, name in zip([xcol, ycol], ['x [nm]', 'y [nm]']):
            if len(item) > 1 or ord(item) < 65:
                indx = int(item) - 1
            else:
                indx = ord(item.lower()) - 97

            try:
                self.df.rename(index=str, columns={self.df.columns[indx]: name}, inplace=True)
            except (IndexError, TypeError):
                messagebox.showerror('Load Data', f'Unable to find column {item}.')
                return False

        return True


    def subsample_data(self, n=500000):
        """Prompts user to sum-sample loaded dataset
        if the number of samples is greater than
        500,000. Sampling is done randomly with a
        set seed for reproducibility between runs.
        """
        if self.XY.shape[0] > n:
            warn = 'Would you like to subsample your dataset for better performance?'
            if messagebox.askyesno('Large Dataset Detected', warn):
                self.XY = self.XY.sample(n, random_state=1).values
            else:
                self.XY = self.XY.values
        else:
            self.XY = self.XY.values


    def ROI(self, update=False, reset=False):
        """Sets the ROI based on matplotlib figure
        extents and allows for updating and resetting
        the ROI.
        """
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
                plot_clusters(self.XY, -np.ones(self.XY.shape[0]), ax=axes[2])
                self.ROI()

        except AttributeError:
            messagebox.showinfo('Reset ROI', 'Please load data first.')


###############################################################################
########################### Computational Functions ###########################
###############################################################################


    def perform_clustering(self):
        """Main clustering function using parameters
        set by GUI sliders and checkbuttons. Computes
        clustering results and provides visualization.
        """
        if not hasattr(self, 'XY'):
            messagebox.showinfo('Run HDBSCAN', 'Please load data first.')
            return

        start = time()
        self.hdb = hdbscan.HDBSCAN(
            core_dist_n_jobs=multiprocessing.cpu_count() - 1,
            gen_min_span_tree=True,
            min_cluster_size=self.min_cluster_size.value,
            min_samples=self.min_samples.value,
            allow_single_cluster=self.allowsingle.get(),
            alpha=self.alpha.scale.get(),
            cluster_selection_method=self.clust_method.get())
        self.hdb.fit(self.XY)

        axes[2].clear()
        self.cluster.configure({'to': len(set(self.hdb.labels_[self.hdb.labels_ != -1])) - 1})
        self.animate(None, force_update=True)
        plot_clusters(self.XY, self.hdb.labels_, ax=axes[2])
        self.cluster_info.config(text=full_cluster_info(self.hdb))
        if not self.hidenotice.get():
            messagebox.showinfo(f'HDBSCAN Results ({time() - start:.1f} seconds)', full_cluster_info(self.hdb))


    def validate(self):
        if not hasattr(self, 'hdb'):
            messagebox.showinfo('Validate', 'Please run HDBSCAN first.')
            return

        if self.validation_mode.get() == 'Relative Validity':
            score = self.hdb.relative_validity_

        if self.validation_mode.get() == 'Silhouette Score':
            score = silhouette_score(self.XY, self.hdb.labels_)
            self.plot_silhouette()
            return

        if self.validation_mode.get() == 'Calinski-Harabasz Index':
            score = calinski_harabasz_score(self.XY, self.hdb.labels_)

        if self.validation_mode.get() == 'Davies-Bouldin Index':
            score = davies_bouldin_score(self.XY, self.hdb.labels_)

        messagebox.showinfo('Validate', f'{self.validation_mode.get()}: {score:.3f}')


    def optimize_params(self):
        """Perform a random hyper parameter search
        across the parameter space defined in the GUI.
        """
        if not hasattr(self, 'XY'):
            messagebox.showinfo('Parameter Search', 'Please load data first.')
            return

        n = simpledialog.askinteger('Parameter Search', 'Enter number of searches:')
        if n:
            param_dist = {"min_cluster_size": randint(self.clustermin.get(), self.clustermax.get()),
                          "min_samples": randint(self.samplemin.get(), self.samplemax.get())}

            self.results = random_search(
                param_dist,
                X=self.XY,
                n=n,
                allowsingle=self.allowsingle.get(),
                alpha=self.alpha.scale.get(),
                silhouette=self.silhouette.get(),
                calinski=self.calinski.get(),
                davies=self.davies.get(),
                method=self.clust_method.get())

            timestamp = pd.Timestamp.now().strftime('%Y-%m-%dT%X')
            self.results.to_csv(f'param-search-results/{self.filename}-{timestamp}.csv', index=False)
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
            except Exception:
                pass


    def draw_region(self):
        """Draw red region box to highlight location
        of current selected cluster (axes[1]).
        """
        ymin, ymax = axes[1].get_ylim()
        xmin, xmax = axes[1].get_xlim()
        xspan = xmax - xmin
        yspan = ymax - ymin

        try:
            self.region.remove()
        except Exception:
            pass

        self.region = Rectangle((xmin, ymin), xspan, yspan,
                                linewidth=1.0, edgecolor=[1,0,0,1],
                                facecolor='none', zorder=3)
        axes[2].add_patch(self.region)


    def draw_probability_marker(self):
        """Draw indicator on probability distribution
        plot based on the current probability threshold.
        """
        marker = axes[0].plot(self.probability.value/100, 0, 'r^', zorder=20)[0]
        marker.set_clip_on(False)


    def plot_param_search(self):
        """Create new window containing parameter
        search heatmap and results table.
        """
        try:
            self.results_window.destroy()
        except Exception:
            pass

        self.results_window = tk.Toplevel(root, background=themebg)
        self.param_fig = Figure(figsize=(8,6), dpi=80, facecolor=themebg)
        self.param_ax = self.param_fig.add_subplot(111)
        self.param_fig.subplots_adjust(right=1.0)

        self.results_plot = UI.PlotFrame(self.results_window, self.param_fig)
        self.results_plot.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        heatmap = self.results.pivot('min_samples', 'min_cluster_size', 'rel_validity')
        sns.heatmap(heatmap, ax=self.param_ax, cmap='RdYlGn')

        var = tk.StringVar()
        scores = self.results.drop(['min_samples', 'min_cluster_size'],
                                   axis=1).columns.values
        ttk.OptionMenu(self.results_window, var, 'Select Plot', *scores,
                   command=self.update_param_plot).pack(side=tk.BOTTOM)

        self.draw_param_table()
        # self.results_window.geometry("1000x600+0+0")


    def plot_silhouette(self):
        """Create new window containing silhouette
        validation plot and score.
        """
        try:
            self.silhouette_window.destroy()
        except Exception:
            pass

        self.silhouette_window = tk.Toplevel(root, background=themebg)
        fig = Figure(dpi=80, facecolor=themebg)
        ax = fig.add_subplot(111)
        UI.PlotFrame(self.silhouette_window, fig).pack(fill=tk.BOTH, expand=True)

        view_silhouette(self.hdb, self.XY, ax)

###############################################################################
############################### Helper Functions ##############################
###############################################################################


    def changed_params(self):
        """Check if any sliders in the GUI have
        been updated, return True if changes occur.
        """
        flag = False
        for key,item in self.current_values.items():
            if self.stored_values[key] != item.value:
                self.stored_values[key] = item.value
                flag=True

        return flag


    def update_param_plot(self, score):
        """Re-draw the parameter search results
        heatmap for different scoring metrics.
        """
        self.param_fig.clear(keep_observers=True)
        self.param_ax = self.param_fig.add_subplot(111)
        heatmap = self.results.pivot('min_samples', 'min_cluster_size', score)
        sns.heatmap(heatmap, ax=self.param_ax, cmap='RdYlGn')
        self.results_plot.canvas.draw()


    def draw_param_table(self):
        """Create interactive parameter search
        results table from results DataFrame.
        """
        subframe = ttk.Frame(self.results_window)
        table = UI.DisplayTable(self.results_window, self.results)
        subframe.pack(side=tk.LEFT, padx=5)

        self.make_header(subframe, "Parameter Search Results")


    def make_header(self, frame, headertext, target=None, state=False):
        """Quick large-font header in the specified
        frame. If target is provided, header will
        include a checkbutton for collapsing or
        expanding the target frame.
        """
        header = ttk.Frame(frame)
        header.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header, text=headertext, anchor='n',
              font=("Courier",16)).pack(side=tk.LEFT, anchor=tk.W, pady=(25,0))
        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X,pady=(5,1))
        # Separator(frame, orient=HORIZONTAL).pack(fill=X,pady=(1,5))

        if target is not None:
            UI.FrameCollapser(header, target, state)


    def make_select(self, frame, text, var):
        """Quick checkbutton with text packed
        in its own frame.
        """
        subframe = ttk.Frame(frame)
        ttk.Checkbutton(subframe, variable=var, text=text).pack(side=tk.BOTTOM, anchor=tk.W)
        subframe.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)


def clearterminal():
    """Clear the terminal window if using
    command line to launch the program.
    """
    if sys.platform.lower() in ['linux', 'darwin']:
        os.system('clear')
    if sys.platform.lower() == 'win32':
        os.system('cls')


###############################################################################
############################# Application Launch ##############################
###############################################################################


# Main function and class calls
if __name__ == "__main__":
    theme = "yaru"
    themefg = "#5c616c"
    themebg = "#f5f6f7"

    # Creating a global figure and axes object for plotting
    f = Figure(figsize=(10,10), dpi=80, facecolor=themebg)
    axes = [f.add_subplot(221), f.add_subplot(222), f.add_subplot(212)]
    f.subplots_adjust(wspace=0.25, top=0.95, right=0.95, bottom=0.08)

    clearterminal()
    root = ThemedTk(theme=theme)
    root.wm_title("HDBSCAN Clustering Utility v0.5.0")
    root.configure(background=themebg)
    root.geometry("1200x1000+0+0")

    app = mainGUI(root).pack(side="top", fill="both", expand=True)
    root.mainloop()
