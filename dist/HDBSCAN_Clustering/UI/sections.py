"""
Author: Philip Ciunkiewicz
"""
from tkinter import *
from tkinter.ttk import *
from ttkwidgets import *


class hdbscanTools(Frame):
    def __init__(self, parent, mainUI):
        Frame.__init__(self, parent)

        controls = Frame(self)
        Button(controls, text="Run HDBSCAN", 
               command=mainUI.perform_clustering).pack(side=RIGHT)
        Button(controls, text="Load Data",
               command=mainUI.load_data).pack(side=LEFT)
        controls.pack()

        Separator(self, orient=HORIZONTAL).pack(fill=X,pady=5)

        scoring = Frame(self)
        scores = ['Relative Validity', 'Silhouette Score',
                  'Calinski-Harabasz Index', 'Davies-Bouldin Index']
        Button(scoring, text="Validate", command=mainUI.validate).pack(side=LEFT)
        OptionMenu(scoring, mainUI.validation_mode, 'Silhouette Score', 
                   *scores).pack(side=RIGHT, anchor=E)
        scoring.pack()

        Separator(self, orient=HORIZONTAL).pack(fill=X,pady=5)

        minclustersize = Frame(self)
        Label(minclustersize, text="Minimum Cluster Size").pack(side=TOP, anchor=W)
        mainUI.min_cluster_size = ScaleEntry(minclustersize, from_=2, to=100, 
                                 scalewidth=200,compound=LEFT)
        mainUI.min_cluster_size.pack(side=BOTTOM)
        mainUI.min_cluster_size._variable.set(50)
        mainUI.min_cluster_size._on_scale(None)
        minclustersize.pack(anchor=W, side=TOP,fill=X)
        
        minsamples = Frame(self)
        Label(minsamples, text="Minimum Samples").pack(side=TOP, anchor=W)
        mainUI.min_samples = ScaleEntry(minsamples, from_=1, to=100, 
                                 scalewidth=200, compound=LEFT)
        mainUI.min_samples.pack(side=BOTTOM)
        mainUI.min_samples._variable.set(5)
        mainUI.min_samples._on_scale(None)
        minsamples.pack(anchor=W, side=TOP, fill=X)

        mainUI.make_select(self, "Hide Results Notification", mainUI.hidenotice)
        
        mainUI.cluster_info = Label(self, text='')
        mainUI.cluster_info.pack(fill=X)


class exploreClusters(Frame):
    def __init__(self, parent, mainUI):
        Frame.__init__(self, parent)
        
        options = Frame(self)
        cluster_n = Frame(options)
        Label(cluster_n, text="Cluster Index").pack(side=TOP, anchor=W)
        mainUI.cluster = ScaleEntry(cluster_n, from_=0, to=1,
                                  scalewidth=200,compound=LEFT)
        mainUI.cluster.pack(side=BOTTOM)
        cluster_n.pack(anchor=W, side=TOP,fill=X)
        
        prob = Frame(options)
        Label(prob, text="Probability Threshold (%)").pack(side=TOP, anchor=W)
        mainUI.probability = ScaleEntry(prob, from_=0, to=99, 
                                 scalewidth=200,compound=LEFT)
        mainUI.probability.pack(side=BOTTOM, fill=X)
        prob.pack(anchor=W, side=TOP, fill=X)
        options.pack(anchor=W, side=TOP, fill=X)

        mainUI.current_values = {
        'cluster' : mainUI.cluster,
        'probability' : mainUI.probability,
        }
        mainUI.stored_values = dict(mainUI.current_values)


class parameterSearch(Frame):
    def __init__(self, parent, mainUI):
        Frame.__init__(self, parent)

        controls = Frame(self)
        mainUI.param_search = Button(controls, text="Optimize Parameters", 
                                   command=mainUI.optimize_params)
        mainUI.param_search.pack(side=RIGHT)
        controls.pack()

        Separator(self, orient=HORIZONTAL).pack(fill=X,pady=5)

        cluster_lims = Frame(self)
        Label(cluster_lims, text="Minimum Cluster range:  ").pack(side=LEFT)
        Entry(cluster_lims, width=4, textvariable=mainUI.clustermax).pack(side=RIGHT, padx=(0,5))
        Label(cluster_lims, text=" - ").pack(side=RIGHT)
        Entry(cluster_lims, width=4, textvariable=mainUI.clustermin).pack(side=RIGHT)
        cluster_lims.pack(anchor=W, fill=X)

        sample_lims = Frame(self)
        Label(sample_lims, text="Minimum Samples range:  ").pack(side=LEFT)
        Entry(sample_lims, width=4, textvariable=mainUI.samplemax).pack(side=RIGHT, padx=(0,5))
        Label(sample_lims, text=" - ").pack(side=RIGHT)
        Entry(sample_lims, width=4, textvariable=mainUI.samplemin).pack(side=RIGHT)
        sample_lims.pack(anchor=W, fill=X)

        mainUI.make_select(self, "Silhouette Score", mainUI.silhouette)
        mainUI.make_select(self, "Calinski-Harabaz Score", mainUI.calinski)
        mainUI.make_select(self, "Davies-Bouldin Score", mainUI.davies)


class ROITools(Frame):
    def __init__(self, parent, mainUI):
        Frame.__init__(self, parent)

        controls = Frame(self)
        mainUI.reset_ROI = Button(controls, text="Reset ROI", 
                                command=lambda: mainUI.ROI(reset=True))
        mainUI.reset_ROI.pack(side=RIGHT)
        mainUI.update_ROI = Button(controls, text="Update ROI", 
                                command=lambda: mainUI.ROI(update=True))
        mainUI.update_ROI.pack(side=RIGHT)
        controls.pack()

        Separator(self, orient=HORIZONTAL).pack(fill=X,pady=5)

        mainUI.ROI_info = Label(self, text='')
        mainUI.ROI_info.pack(fill=X)
        mainUI.ROI()


class advancedTools(Frame):
    def __init__(self, parent, mainUI):
        Frame.__init__(self, parent)

        options = Frame(self)
        alph = Frame(options)
        Label(alph, text="Alpha").pack(side=TOP, anchor=W)
        mainUI.alpha = TickScale(alph, orient='horizontal', from_=0.5, 
                                 to=2.0, tickinterval=0, resolution=0.001,
                                 labelpos='w')
        mainUI.alpha.scale.set(1.0)
        mainUI.alpha.pack(side=BOTTOM, fill=X)
        alph.pack(anchor=W, side=TOP, fill=X)

        method = Frame(options)
        Label(method, text="Cluster Selection Method:").pack(side=LEFT, anchor=W)
        OptionMenu(method, mainUI.clust_method, 'eom', 
                   *['eom', 'leaf']).pack(side=RIGHT, anchor=E)
        method.pack(anchor=W, side=TOP, fill=X)
        options.pack(anchor=W, side=TOP, fill=X)

        mainUI.make_select(self, "Allow Single Cluster", mainUI.allowsingle)
