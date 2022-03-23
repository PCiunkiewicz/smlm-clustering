"""
Author: Philip Ciunkiewicz
"""
import tkinter as tk
from tkinter import ttk
import ttkwidgets as ttkw


class HDBScanTools(ttk.Frame):
    def __init__(self, parent, mainUI):
        super().__init__(parent)

        controls = ttk.Frame(self)
        ttk.Button(controls, text="Run HDBSCAN",
               command=mainUI.perform_clustering).pack(side=tk.RIGHT)
        ttk.Button(controls, text="Load Data",
               command=mainUI.load_data).pack(side=tk.LEFT)
        controls.pack()

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5)

        scoring = ttk.Frame(self)
        scores = ['Relative Validity', 'Silhouette Score',
                  'Calinski-Harabasz Index', 'Davies-Bouldin Index']
        ttk.Button(scoring, text="Validate", command=mainUI.validate).pack(side=tk.LEFT)
        ttk.OptionMenu(scoring, mainUI.validation_mode, 'Silhouette Score',
                   *scores).pack(side=tk.RIGHT, anchor=tk.E)
        scoring.pack()

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5)

        minclustersize = ttk.Frame(self)
        ttk.Label(minclustersize, text="Minimum Cluster Size").pack(side=tk.TOP, anchor=tk.W)
        mainUI.min_cluster_size = ttkw.ScaleEntry(minclustersize, from_=2, to=100,
                                 scalewidth=250,compound=tk.LEFT)
        mainUI.min_cluster_size.pack(side=tk.BOTTOM)
        mainUI.min_cluster_size._variable.set(50)
        mainUI.min_cluster_size._on_scale(None)
        minclustersize.pack(anchor=tk.W, side=tk.TOP,fill=tk.X)

        minsamples = ttk.Frame(self)
        ttk.Label(minsamples, text="Minimum Samples").pack(side=tk.TOP, anchor=tk.W)
        mainUI.min_samples = ttkw.ScaleEntry(minsamples, from_=1, to=100,
                                 scalewidth=250, compound=tk.LEFT)
        mainUI.min_samples.pack(side=tk.BOTTOM)
        mainUI.min_samples._variable.set(5)
        mainUI.min_samples._on_scale(None)
        minsamples.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)

        mainUI.make_select(self, "Hide Results Notification", mainUI.hidenotice)

        mainUI.cluster_info = ttk.Label(self, text='')
        mainUI.cluster_info.pack(fill=tk.X)


class ExploreClusters(ttk.Frame):
    def __init__(self, parent, mainUI):
        super().__init__(parent)

        options = ttk.Frame(self)
        cluster_n = ttk.Frame(options)
        ttk.Label(cluster_n, text="Cluster Index").pack(side=tk.TOP, anchor=tk.W)
        mainUI.cluster = ttkw.ScaleEntry(cluster_n, from_=0, to=1,
                                  scalewidth=250,compound=tk.LEFT)
        mainUI.cluster.pack(side=tk.BOTTOM)
        cluster_n.pack(anchor=tk.W, side=tk.TOP,fill=tk.X)

        prob = ttk.Frame(options)
        ttk.Label(prob, text="Probability Threshold (%)").pack(side=tk.TOP, anchor=tk.W)
        mainUI.probability = ttkw.ScaleEntry(prob, from_=0, to=99,
                                 scalewidth=250,compound=tk.LEFT)
        mainUI.probability.pack(side=tk.BOTTOM, fill=tk.X)
        prob.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)
        options.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)

        mainUI.current_values = {
        'cluster' : mainUI.cluster,
        'probability' : mainUI.probability,
        }
        mainUI.stored_values = dict(mainUI.current_values)


class ParameterSearch(ttk.Frame):
    def __init__(self, parent, mainUI):
        super().__init__(parent)

        controls = ttk.Frame(self)
        mainUI.param_search = ttk.Button(controls, text="Optimize Parameters",
                                   command=mainUI.optimize_params)
        mainUI.param_search.pack(side=tk.RIGHT)
        controls.pack()

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5)

        cluster_lims = ttk.Frame(self)
        ttk.Label(cluster_lims, text="Minimum Cluster range:  ").pack(side=tk.LEFT)
        ttk.Entry(cluster_lims, width=4, textvariable=mainUI.clustermax).pack(side=tk.RIGHT, padx=(0,5))
        ttk.Label(cluster_lims, text=" - ").pack(side=tk.RIGHT)
        ttk.Entry(cluster_lims, width=4, textvariable=mainUI.clustermin).pack(side=tk.RIGHT)
        cluster_lims.pack(anchor=tk.W, fill=tk.X)

        sample_lims = ttk.Frame(self)
        ttk.Label(sample_lims, text="Minimum Samples range:  ").pack(side=tk.LEFT)
        ttk.Entry(sample_lims, width=4, textvariable=mainUI.samplemax).pack(side=tk.RIGHT, padx=(0,5))
        ttk.Label(sample_lims, text=" - ").pack(side=tk.RIGHT)
        ttk.Entry(sample_lims, width=4, textvariable=mainUI.samplemin).pack(side=tk.RIGHT)
        sample_lims.pack(anchor=tk.W, fill=tk.X)

        mainUI.make_select(self, "Silhouette Score", mainUI.silhouette)
        mainUI.make_select(self, "Calinski-Harabasz Score", mainUI.calinski)
        mainUI.make_select(self, "Davies-Bouldin Score", mainUI.davies)


class ROITools(ttk.Frame):
    def __init__(self, parent, mainUI):
        super().__init__(parent)

        controls = ttk.Frame(self)
        mainUI.reset_ROI = ttk.Button(controls, text="Reset ROI",
                                command=lambda: mainUI.ROI(reset=True))
        mainUI.reset_ROI.pack(side=tk.RIGHT)
        mainUI.update_ROI = ttk.Button(controls, text="Update ROI",
                                command=lambda: mainUI.ROI(update=True))
        mainUI.update_ROI.pack(side=tk.RIGHT)
        controls.pack()

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5)

        mainUI.ROI_info = ttk.Label(self, text='')
        mainUI.ROI_info.pack(fill=tk.X)
        mainUI.ROI()


class AdvancedTools(ttk.Frame):
    def __init__(self, parent, mainUI):
        super().__init__(parent)

        options = ttk.Frame(self)
        alph = ttk.Frame(options)
        ttk.Label(alph, text="Alpha").pack(side=tk.TOP, anchor=tk.W)
        mainUI.alpha = ttkw.TickScale(alph, orient='horizontal', from_=0.5,
                                 to=2.0, tickinterval=0, resolution=0.001,
                                 labelpos='w')
        mainUI.alpha.scale.set(1.0)
        mainUI.alpha.pack(side=tk.BOTTOM, fill=tk.X)
        alph.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)

        method = ttk.Frame(options)
        ttk.Label(method, text="Cluster Selection Method:").pack(side=tk.LEFT, anchor=tk.W)
        ttk.OptionMenu(method, mainUI.clust_method, 'eom',
                   *['eom', 'leaf']).pack(side=tk.RIGHT, anchor=tk.E)
        method.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)
        options.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)

        mainUI.make_select(self, "Allow Single Cluster", mainUI.allowsingle)
