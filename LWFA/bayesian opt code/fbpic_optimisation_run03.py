# -------
# Imports
# -------
import os
import numba
#import math
import numpy as np
import matplotlib.pyplot as plt

from skopt import gp_minimize, callbacks, load
from skopt.benchmarks import bench2
from skopt.utils import cook_initial_point_generator
from skopt.space import Real
from skopt.plots import plot_evaluations, plot_objective, plot_convergence, plot_gaussian_process

# Function to optimise, placeholder for simulation
def lwfa_simulation(x):
	# A benchmark function for test purposes.
	# f(x) = x ** 2 if x < 0
	# 	(x-5) ** 2 - 5 otherwise.
	# It has a global minima with f(x*) = -5 at x* = 5.
	return bench2(x)

def bayesian_optimisation():
	# Where to save optimisation checkpoints
	opt_save_path = "./"
	checkpoint_name = "checkpoint_run03.pkl"

	checkpoint_saver = callbacks.CheckpointSaver(checkpoint_name, compress=9)
	lhs_maximin = cook_initial_point_generator("lhs", criterion="maximin")

	x0 = None
	y0 = None
	if os.path.exists(opt_save_path + checkpoint_name):
		print('> Loading optimisation checkpoint')
		res = load(opt_save_path + checkpoint_name)
		x0 = res.x_iters
		y0 = res.func_vals
	else:
		print('> Starting new optimisation')

	res = gp_minimize(lwfa_simulation,
					  [Real(1.0, 5.0, transform='identity', name='a0')],
					  x0=x0, y0=y0,
					  xi=0.0001,
					  kappa=0.001,
					  acq_func='EI', acq_optimizer='sampling',
					  n_calls=25, n_initial_points=10, initial_point_generator=lhs_maximin,
					  callback=[checkpoint_saver],
					  verbose = True,
					  noise = 1e-10,
					  random_state = 42)

	return res

def view_best():
	print("LOADING ACTUAL BEST RESULTS FROM LASER OPTIMISATION\n")
	opt_save_path = "../best optimisation runs/"
	res = load(opt_save_path + 'run03_checkpoint.pkl')
	for i in range(len(res.func_vals)):
		if(res.func_vals[i]<15.0):
			print("Fitness Score:", res.func_vals[i])
			print("a0:", res.x_iters[i])
			print("-----")

if __name__ == '__main__':
	res = bayesian_optimisation()
	print("\nResult found", res.fun, "using parameters",res.x)
	print("-----")
	view_best()

	a = plot_convergence(res)
	plt.show()
	b = plot_gaussian_process(res)
	plt.show()