"""
Author: Philip Ciunkiewicz
"""
from tkinter import *
from tkinter.ttk import *
from .sections import *


class controlsFrame(Frame):
    """Main controls frame for clustering GUI.
    """
    def __init__(self, parent, mainUI):
        Frame.__init__(self, parent)

        Separator(self, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=(10,10))
        Separator(self, orient=VERTICAL).pack(side=RIGHT, fill=Y, padx=(10,0))

        container = Frame(self)
        Frame(container).pack()
        self.hdbscanTools = hdbscanTools(container, mainUI)
        mainUI.make_header(self, "HDBSCAN Tools", self.hdbscanTools, state=True)
        container.pack(fill=X)
        Separator(self, orient=HORIZONTAL).pack(fill=X, pady=(1,5))
        
        container = Frame(self)
        Frame(container).pack()
        self.exploreClusters = exploreClusters(container, mainUI)
        mainUI.make_header(self, "Explore Clusters", self.exploreClusters)
        container.pack(fill=X)
        Separator(self, orient=HORIZONTAL).pack(fill=X, pady=(1,5))
        
        container = Frame(self)
        Frame(container).pack()
        self.parameterSearch =  parameterSearch(container, mainUI)
        mainUI.make_header(self, "Parameter Search", self.parameterSearch)
        container.pack(fill=X)
        Separator(self, orient=HORIZONTAL).pack(fill=X, pady=(1,5))
        
        container = Frame(self)
        Frame(container).pack()
        self.ROITools = ROITools(container, mainUI)
        mainUI.make_header(self, "ROI Tools", self.ROITools)
        container.pack(fill=X)
        Separator(self, orient=HORIZONTAL).pack(fill=X, pady=(1,5))
        
        container = Frame(self)
        Frame(container).pack()
        self.advancedTools = advancedTools(container, mainUI)
        mainUI.make_header(self, "Advanced Tools", self.advancedTools)
        container.pack(fill=X)
        Separator(self, orient=HORIZONTAL).pack(fill=X, pady=(1,5))
