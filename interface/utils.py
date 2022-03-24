"""
Author: Philip Ciunkiewicz
"""
import os
import sys
import inspect
import tkinter as tk
from tkinter import ttk

from .framecollapser import FrameCollapser


THEME = 'yaru'
THEMEFG = '#5c616c'
THEMEBG = '#f5f6f7'


def make_header(frame, text, target=None, state=False):
    """Quick large-font header in the specified
    frame. If target is provided, header will
    include a checkbutton for collapsing or
    expanding the target frame.
    """
    header = ttk.Frame(frame)
    header.pack(side=tk.TOP, fill=tk.X)
    ttk.Label(header, text=text, anchor='n', font=('Courier', 20)).pack(side=tk.LEFT, anchor=tk.W, pady=(25,0))
    ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X,pady=(5,1))
    # Separator(frame, orient=HORIZONTAL).pack(fill=X,pady=(1,5))

    if target is not None:
        FrameCollapser(header, target, state)


def make_select(frame, text, var):
    """Quick checkbutton with text packed
    in its own frame.
    """
    subframe = ttk.Frame(frame)
    ttk.Checkbutton(subframe, variable=var, text=text).pack(side=tk.BOTTOM, anchor=tk.W)
    subframe.pack(anchor=tk.W, side=tk.TOP, fill=tk.X)


def convert_columns(df):
    """Rename columns if they do not follow
    ThunderSTORM convention.
    """
    prompt = inspect.cleandoc(
        """Unable to automatically identify columns.
        Please provide the column index / letter for
        each of the following options.
        """)
    tk.messagebox.showinfo('Load Data', prompt)
    xcol = tk.simpledialog.askstring('Load Data', 'Identify X column.')
    ycol = tk.simpledialog.askstring('Load Data', 'Identify Y column.')
    # intensitycol = simpledialog.askstring('Load Data', 'Identify intensity column.')

    for item, name in zip([xcol, ycol], ['x [nm]', 'y [nm]']):
        if len(item) > 1 or ord(item) < 65:
            indx = int(item) - 1
        else:
            indx = ord(item.lower()) - 97

        try:
            df.rename(index=str, columns={df.columns[indx]: name}, inplace=True)
        except (IndexError, TypeError):
            tk.messagebox.showerror('Load Data', f'Unable to find column {item}.')
            return False

    return True


def clear_terminal():
    """Clear the terminal window if using
    command line to launch the program.
    """
    if sys.platform.lower() in ['linux', 'darwin']:
        os.system('clear')
    if sys.platform.lower() == 'win32':
        os.system('cls')
