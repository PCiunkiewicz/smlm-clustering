"""
Author: Philip Ciunkiewicz
"""
import tkinter as tk
from tkinter import ttk


class frameCollapser(ttk.Checkbutton):
	"""Checkbutton for collapsing and
	expanding frames.
	"""
	def __init__(self, parent, target, state):
		var = tk.BooleanVar()
		super().__init__(parent, command=self.collapse, variable=var)
		self.target = target
		if state:
			self.collapse()
			var.set(True)

		self.pack(side=tk.RIGHT, anchor=tk.SE)

	def collapse(self):
		if self.target.winfo_ismapped():
			self.target.pack_forget()

		else:
			self.target.pack(fill=tk.X)
