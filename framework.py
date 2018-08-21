#------------------------------------------------------------------------------
#						02 Oct 2017 Thrusday 08:50 AM
#------------------------------------------------------------------------------

import tkinter as tk

#CREATE GUI FRAMEWORK 
class GUIFramework(object):
	menuitems = None
	
	def __init__(self, root):
		self.root = root
		if self.menuitems is not None:
			self.build_menu()
		
		#CREATING MENU BUILDER
	
	def build_menu(self):
		self.menubar = tk.Menu(self.root)
	
		for v in self.menuitems:
			menu = tk.Menu(self.menubar, tearoff=0)
			label, items = v.split('-')
			items = map(str.strip, items.split(','))
			for item in items:
				self.__add_menu_command(menu, item)
			self.menubar.add_cascade(label=label, menu=menu)
		self.root.config(menu=self.menubar)
		
	def __add_menu_command(self, menu, item):
		if item == 'Sep':
			menu.add_separator()
		else:
			name, acc, cmd = item.split('/')
			try:
				underline = name.index('&')
				name = name.replace('&', '', 1)
			except ValueError:
				underline = None
			menu.add_command(label=name, underline=underline, accelerator=acc, command=eval(cmd))

