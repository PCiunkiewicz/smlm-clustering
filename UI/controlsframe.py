"""
Author: Philip Ciunkiewicz
"""
import tkinter as tk
from tkinter import ttk
from .sections import HDBScanTools, ExploreClusters, ParameterSearch, ROITools, AdvancedTools


class ControlsFrame(ttk.Frame):
    """Main controls frame for clustering GUI.
    """
    def __init__(self, parent, mainUI):
        super().__init__(parent)

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=(10,10))
        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.RIGHT, fill=tk.Y, padx=(10,0))

        container = ttk.Frame(self)
        ttk.Frame(container).pack()
        self.hdbscanTools = HDBScanTools(container, mainUI)
        mainUI.make_header(self, "HDBSCAN")
        self.hdbscanTools.pack(fill=tk.X)
        container.pack(fill=tk.X)
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(1,5))

        container = ttk.Frame(self)
        ttk.Frame(container).pack()
        self.exploreClusters = ExploreClusters(container, mainUI)
        mainUI.make_header(self, "Explore Clusters", self.exploreClusters)
        container.pack(fill=tk.X)
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(1,5))

        container = ttk.Frame(self)
        ttk.Frame(container).pack()
        self.parameterSearch =  ParameterSearch(container, mainUI)
        mainUI.make_header(self, "Parameter Search", self.parameterSearch)
        container.pack(fill=tk.X)
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(1,5))

        container = ttk.Frame(self)
        ttk.Frame(container).pack()
        self.ROITools = ROITools(container, mainUI)
        mainUI.make_header(self, "ROI", self.ROITools)
        container.pack(fill=tk.X)
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(1,5))

        container = ttk.Frame(self)
        ttk.Frame(container).pack()
        self.advancedTools = AdvancedTools(container, mainUI)
        mainUI.make_header(self, "Advanced", self.advancedTools)
        container.pack(fill=tk.X)
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(1,5))
