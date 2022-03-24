"""
Author: Philip Ciunkiewicz
"""
import tkinter as tk
from tkinter import ttk
import ttkwidgets as ttkw

from .utils import make_select


class HDBScanTools(ttk.Frame):
    def __init__(self, parent, main_gui):
        super().__init__(parent)

        controls = ttk.Frame(self)
        ttk.Button(controls, text='Run HDBSCAN', command=main_gui.perform_clustering).pack(side=tk.RIGHT)
        ttk.Button(controls, text='Load Data', command=main_gui.load_data).pack(side=tk.LEFT)
        controls.pack()

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5)

        scoring = ttk.Frame(self)
        scores = [
            'Relative Validity',
            'Silhouette Score',
            'Calinski-Harabasz Index',
            'Davies-Bouldin Index'
        ]
        ttk.Button(scoring, text='Validate', command=main_gui.validate).pack(side=tk.LEFT)
        ttk.OptionMenu(scoring, main_gui.validation_mode, 'Silhouette Score', *scores).pack(side=tk.RIGHT, anchor=tk.E)
        scoring.pack()

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5)

        minclustersize = ttk.Frame(self)
        ttk.Label(minclustersize, text='Minimum Cluster Size').pack(side=tk.TOP, anchor=tk.W)
        main_gui.min_cluster_size = ttkw.ScaleEntry(minclustersize, from_=2, to=100, scalewidth=250,compound=tk.LEFT)
        main_gui.min_cluster_size.pack(side=tk.BOTTOM)
        main_gui.min_cluster_size._variable.set(50)
        main_gui.min_cluster_size._on_scale(None)
        minclustersize.pack(anchor=tk.W, side=tk.TOP,fill=tk.X)

        minsamples = ttk.Frame(self)
        ttk.Label(minsamples, text='Minimum Samples').pack(side=tk.TOP, anchor=tk.W)
        main_gui.min_samples = ttkw.ScaleEntry(minsamples, from_=1, to=100, scalewidth=250, compound=tk.LEFT)
        main_gui.min_samples.pack(side=tk.BOTTOM)
        main_gui.min_samples._variable.set(5)
        main_gui.min_samples._on_scale(None)
        minsamples.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)

        make_select(self, 'Hide Results Notification', main_gui.hidenotice)

        main_gui.cluster_info = ttk.Label(self, text='')
        main_gui.cluster_info.pack(fill=tk.X)


class ExploreClusters(ttk.Frame):
    def __init__(self, parent, main_gui):
        super().__init__(parent)

        options = ttk.Frame(self)
        cluster_n = ttk.Frame(options)
        ttk.Label(cluster_n, text='Cluster Index').pack(side=tk.TOP, anchor=tk.W)
        main_gui.cluster = ttkw.ScaleEntry(cluster_n, from_=0, to=1, scalewidth=250,compound=tk.LEFT)
        main_gui.cluster.pack(side=tk.BOTTOM)
        cluster_n.pack(anchor=tk.W, side=tk.TOP,fill=tk.X)

        prob = ttk.Frame(options)
        ttk.Label(prob, text='Probability Threshold (%)').pack(side=tk.TOP, anchor=tk.W)
        main_gui.probability = ttkw.ScaleEntry(prob, from_=0, to=99, scalewidth=250,compound=tk.LEFT)
        main_gui.probability.pack(side=tk.BOTTOM, fill=tk.X)
        prob.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)
        options.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)

        main_gui.current_values = {
            'cluster' : main_gui.cluster,
            'probability' : main_gui.probability,
        }
        main_gui.stored_values = dict(main_gui.current_values)


class ParameterSearch(ttk.Frame):
    def __init__(self, parent, main_gui):
        super().__init__(parent)

        controls = ttk.Frame(self)
        main_gui.param_search = ttk.Button(controls, text='Optimize Parameters', command=main_gui.optimize_params)
        main_gui.param_search.pack(side=tk.RIGHT)
        controls.pack()

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5)

        cluster_lims = ttk.Frame(self)
        ttk.Label(cluster_lims, text='Minimum Cluster range:  ').pack(side=tk.LEFT)
        ttk.Entry(cluster_lims, width=4, textvariable=main_gui.clustermax).pack(side=tk.RIGHT, padx=(0,5))
        ttk.Label(cluster_lims, text=' - ').pack(side=tk.RIGHT)
        ttk.Entry(cluster_lims, width=4, textvariable=main_gui.clustermin).pack(side=tk.RIGHT)
        cluster_lims.pack(anchor=tk.W, fill=tk.X)

        sample_lims = ttk.Frame(self)
        ttk.Label(sample_lims, text='Minimum Samples range:  ').pack(side=tk.LEFT)
        ttk.Entry(sample_lims, width=4, textvariable=main_gui.samplemax).pack(side=tk.RIGHT, padx=(0,5))
        ttk.Label(sample_lims, text=' - ').pack(side=tk.RIGHT)
        ttk.Entry(sample_lims, width=4, textvariable=main_gui.samplemin).pack(side=tk.RIGHT)
        sample_lims.pack(anchor=tk.W, fill=tk.X)

        make_select(self, 'Silhouette Score', main_gui.silhouette)
        make_select(self, 'Calinski-Harabasz Score', main_gui.calinski)
        make_select(self, 'Davies-Bouldin Score', main_gui.davies)


class ROITools(ttk.Frame):
    def __init__(self, parent, main_gui):
        super().__init__(parent)

        controls = ttk.Frame(self)
        main_gui.reset_roi = ttk.Button(controls, text='Reset ROI', command=lambda: main_gui.roi(reset=True))
        main_gui.reset_roi.pack(side=tk.RIGHT)
        main_gui.update_roi = ttk.Button(controls, text='Update ROI', command=lambda: main_gui.roi(update=True))
        main_gui.update_roi.pack(side=tk.RIGHT)
        controls.pack()

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X,pady=5)

        main_gui.roi_info = ttk.Label(self, text='')
        main_gui.roi_info.pack(fill=tk.X)
        main_gui.roi()


class AdvancedTools(ttk.Frame):
    def __init__(self, parent, main_gui):
        super().__init__(parent)

        options = ttk.Frame(self)
        alpha = ttk.Frame(options)
        ttk.Label(alpha, text='Alpha').pack(side=tk.TOP, anchor=tk.W)
        main_gui.alpha = ttkw.TickScale(
            alpha,
            orient='horizontal',
            from_=0.5,
            to=2.0,
            tickinterval=0,
            resolution=0.001,
            labelpos='w')
        main_gui.alpha.scale.set(1.0)
        main_gui.alpha.pack(side=tk.BOTTOM, fill=tk.X)
        alpha.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)

        method = ttk.Frame(options)
        ttk.Label(method, text='Cluster Selection Method:').pack(side=tk.LEFT, anchor=tk.W)
        ttk.OptionMenu(method, main_gui.clust_method, 'eom', *['eom', 'leaf']).pack(side=tk.RIGHT, anchor=tk.E)
        method.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)
        options.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)

        make_select(self, 'Allow Single Cluster', main_gui.allowsingle)
