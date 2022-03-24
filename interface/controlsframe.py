"""
Author: Philip Ciunkiewicz
"""
import tkinter as tk
from tkinter import ttk
from .sections import HDBScanTools, ExploreClusters, ParameterSearch, ROITools, AdvancedTools

from .utils import make_header


class ControlsFrame(ttk.Frame):
    """Main controls frame for clustering GUI.
    """
    def __init__(self, parent, main_gui):
        super().__init__(parent)
        self.main_gui = main_gui

        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=(10,10))
        ttk.Separator(self, orient=tk.VERTICAL).pack(side=tk.RIGHT, fill=tk.Y, padx=(10,0))

        self._section_factory('HDBSCAN', HDBScanTools, collapse=False)
        self._section_factory('Explore Clusters', ExploreClusters, collapse=False)
        self._section_factory('ROI', ROITools, collapse=False)
        self._section_factory('Parameter Search', ParameterSearch)
        self._section_factory('Advanced', AdvancedTools)

    def _section_factory(self, name, section_class, collapse=True):
        container = ttk.Frame(self)
        ttk.Frame(container).pack()

        if collapse:
            make_header(self, name, section_class(container, self.main_gui))
        else:
            make_header(self, name)
            section_class(container, self.main_gui).pack(fill=tk.X)

        container.pack(fill=tk.X)
        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(1,5))
