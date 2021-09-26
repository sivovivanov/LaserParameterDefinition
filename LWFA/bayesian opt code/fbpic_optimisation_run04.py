# -------
# Imports
# -------
import os
import numba
#import math
import numpy as np
import matplotlib.pyplot as plt

from skopt import gp_minimize, callbacks, load
from skopt.benchmarks import branin
from skopt.utils import cook_initial_point_generator
from skopt.space import Real
from skopt.plots import plot_evaluations, plot_objective, plot_convergence, plot_gaussian_process

# Function to optimise, placeholder for simulation
def lwfa_simulation(x):
	# Branin-Hoo function is defined on the square.
	# It has three minima with f(x*) = 0.397887 at x* = (-pi, 12.275), (+pi, 2.275), and (9.42478, 2.475).
	return branin(x)

def bayesian_optimisation():
    # Where to save optimisation checkpoints
    checkpoint_name = 'checkpoint_run04.pkl'

    checkpoint_saver = callbacks.CheckpointSaver(checkpoint_name, compress=9)
    lhs_maximin = cook_initial_point_generator("lhs", criterion="maximin")

    x0, y0 = None, None
    if  os.path.exists(checkpoint_name):
        print('> Loading optimisation checkpoint')
        res = load(checkpoint_name)
        x0 = res.x_iters
        y0 = res.func_vals
    else:
        print('> Starting new optimisation')

    res = gp_minimize(lwfa_simulation,
                      [Real(1.0, 5.0, transform='identity', name='a0'),
                       Real(5e-6, 15e-6, transform='identity', name='w0')],
                      x0=x0, y0=y0,
                      xi=0.000001,
                      kappa=0.001,
                      acq_func='EI', acq_optimizer='sampling',
                      n_calls=50, n_initial_points=20, initial_point_generator=lhs_maximin,
                      callback=[checkpoint_saver],
                      verbose = True,
                      noise = 1e-10,
                      random_state = 42)

    return res
	
def view_best():
	print("LOADING ACTUAL BEST RESULTS FROM LASER OPTIMISATION\n")
	opt_save_path = "../best optimisation runs/"
	res = load(opt_save_path + 'run04_checkpoint.pkl')
	for i in range(len(res.func_vals)):
		if(res.func_vals[i]<14.0):
			print("Fitness Score:", res.func_vals[i])
			print("a0 and w0:", res.x_iters[i])
			print("-----")

if __name__ == '__main__':
	res = bayesian_optimisation()
	print("\nResult found", res.fun, "using parameters",res.x)
	print("-----")
	view_best()

	a = plot_convergence(res)
	plt.show()