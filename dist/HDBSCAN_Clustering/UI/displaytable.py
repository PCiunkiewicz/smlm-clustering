"""
Author: Philip Ciunkiewicz
"""
from tkinter import *
from tkinter.ttk import *
from ttkwidgets import Table


class displayTable(Frame):
    """Frame for displaying a Pandas DataFrame as
    an interactive table with ttkwidgets.
    """
    def __init__(self, parent, data):
        Frame.__init__(self, parent)
        columns = data.columns.values
        table = Table(self, columns=columns, drag_cols=True,
                      drag_rows=False, height=20)

        for indx, col in enumerate(columns):
            table.heading(indx, text=col)
            table.column(indx, minwidth=50, width=100, stretch=False, type=float)

        for i, x in enumerate(data.values):
            table.insert('', 'end', iid=i, values=tuple(x))

        sx = Scrollbar(self, orient='horizontal', command=table.xview)
        sy = Scrollbar(self, orient='vertical', command=table.yview)
        table.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)

        sx.pack(side=BOTTOM, anchor=S, fill=X)
        sy.pack(side=RIGHT, anchor=E, fill=Y)
        table.pack(side=BOTTOM, expand=True, fill=BOTH)

        self.pack(side=BOTTOM, expand=True, fill=BOTH)
