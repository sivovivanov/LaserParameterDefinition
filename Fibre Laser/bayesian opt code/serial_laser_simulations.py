#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from skopt import gp_minimize, callbacks, load
from skopt.utils import cook_initial_point_generator
from skopt.space import Real
from skopt.plots import plot_evaluations, plot_objective, plot_convergence

# This function called the simulation
def simulate(x):
	fitness = ((x[0] / x[1]) + (x[2] * x[3])) + x[4]
	return fitness

def bayesian_optimisation():
	# Where to save optimisation checkpoints
	opt_save_path = "./"
	checkpoint_name = "checkpoint.pkl"

	checkpoint_saver = callbacks.CheckpointSaver(opt_save_path + checkpoint_name, compress=9)
	lhs_maximin = cook_initial_point_generator("lhs", criterion="maximin")

	x0, y0 = None, None
	if os.path.exists(opt_save_path + checkpoint_name):
		print('> Loading optimisation checkpoint')
		res = load(opt_save_path + checkpoint_name)
		x0 = res.x_iters
		y0 = res.func_vals
	else:
		print('> Starting new optimisation')

	res = gp_minimize(simulate,
					[Real(8300.0, 8800.0, transform="identity", name="Doping Concentration"),
					 Real(2.0e+00, 2.5e+00, transform="identity", name="HWP"),
					 Real(-1.5e+00, -1.0e+00, transform="identity", name="QWP1"),
					 Real(-2.1e+00, -1.5e+00, transform="identity", name="QWP2"),
					 Real(5.0e-15, 10.0e-15, transform='identity', name="Group Velocity Mismatch")],
					x0 = x0, y0 = y0,
					xi = -10000000000, kappa = 0.001,
					acq_func = "EI", acq_optimizer = "sampling",
					n_calls = 25, n_initial_points = 10,
					initial_point_generator =lhs_maximin, callback = [checkpoint_saver],
					verbose = True, noise = 1e-10,
					random_state = 42)

	return res

def view_best():
	print("LOADING ACTUAL BEST RESULTS FROM LASER OPTIMISATION\n")
	opt_save_path = "../best optimisation runs/"
	res = load(opt_save_path + 'checkpoint0_39.pkl')
	for i in range(len(res.func_vals)):
		if(res.func_vals[i]<0.5):
			print("Fitness Score:", res.func_vals[i])
			print("Parameters:", res.x_iters[i])
			print("-----")

if __name__ == '__main__':
	res = bayesian_optimisation()
	print("\nResult found", res.fun, "using parameters",res.x)
	print("-----")
	view_best()

	plot_evaluations(res)
	plt.show()

	plot_objective(res, n_samples=50)
	plt.show()

	plot_convergence(res)
	plt.show()