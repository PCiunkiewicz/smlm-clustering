"""
Author: Philip Ciunkiewicz
"""
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from tkinter import *
from tkinter.ttk import *


class plotFrame(Frame):
    """Frame for the plot to be animated within, 
    including TKinter drawing canvas and matplotlib toolbar.
    """
    def __init__(self, parent, fig):
        Frame.__init__(self, parent)

        self.canvas = FigureCanvasTkAgg(fig, self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=LEFT, fill=BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, self)
        toolbar.config(background="#f5f6f7")
        toolbar._message_label.config(background="#f5f6f7")
        toolbar.update()
        self.canvas._tkcanvas.pack(side=TOP, fill=BOTH, expand=True)
