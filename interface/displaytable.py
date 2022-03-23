"""
Author: Philip Ciunkiewicz
"""
import tkinter as tk
from tkinter import ttk
from ttkwidgets import Table


class DisplayTable(ttk.Frame):
    """Frame for displaying a Pandas DataFrame as
    an interactive table with ttkwidgets.
    """
    def __init__(self, parent, data):
        super().__init__(parent)
        columns = data.columns.values
        table = Table(self, columns=columns, drag_cols=False, drag_rows=False, height=20)

        for indx, col in enumerate(columns):
            table.heading(indx, text=col)
            table.column(indx, minwidth=50, width=100, stretch=False, type=float)

        for i, x in enumerate(data.values):
            table.insert('', 'end', iid=i, values=tuple(x))

        sx = ttk.Scrollbar(self, orient='horizontal', command=table.xview)
        sy = ttk.Scrollbar(self, orient='vertical', command=table.yview)
        table.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)

        sx.pack(side=tk.BOTTOM, anchor=tk.S, fill=tk.X)
        sy.pack(side=tk.RIGHT, anchor=tk.E, fill=tk.Y)
        table.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH)

        self.pack(side=tk.BOTTOM, expand=True, fill=tk.BOTH)
