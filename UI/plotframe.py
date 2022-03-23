"""
Author: Philip Ciunkiewicz
"""
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


class PlotFrame(ttk.Frame):
    """Frame for the plot to be animated within,
    including TKinter drawing canvas and matplotlib toolbar.
    """
    def __init__(self, parent, fig):
        super().__init__(parent)

        self.canvas = FigureCanvasTkAgg(fig, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.config(background="#f5f6f7")
        toolbar._message_label.config(background="#f5f6f7")
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
