"""
Author: Philip Ciunkiewicz
"""
from tkinter import *
from tkinter.ttk import *
from .sections import *


class frameCollapser(Checkbutton):
	"""Checkbutton for collapsing and
	expanding frames.
	"""
	def __init__(self, parent, target, state):
		var = BooleanVar()
		Checkbutton.__init__(self, parent, command=self.collapse, variable=var)
		self.target = target
		if state:
			self.collapse()
			var.set(True)

		self.pack(side=RIGHT, anchor=SE)

	def collapse(self):
		if self.target.winfo_ismapped():
			self.target.pack_forget()

		else:
			self.target.pack(fill=X)
