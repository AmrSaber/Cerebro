
import os
__cwd__ = ""

def set_cwd(x):
	global __cwd__
	__cwd__ = x
	if not os.path.isdir(__cwd__): 
		raise Exception("This directory doesn't exist.")
