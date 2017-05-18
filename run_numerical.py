from __future__ import division
import signal
import multiprocessing as mp
import numpy as np
import pickle
import click
from math import pi

# Some command line arguments aren't meant to be negative
class PositiveFloatC(click.ParamType):
	name = 'float'

	def __init__(self, zero_allowed = False):
		click.ParamType.__init__(self)
		self.zero_allowed = zero_allowed

	def convert(self, value, param, ctx):
		try:
			ret = float(value)
			assert(ret >= 0 and (self.zero_allowed or ret > 0))
		except (ValueError, AssertionError):
			self.fail("Parameter '{}' needs to be a positive float".format(param))
		
		return ret

PositiveFloat = PositiveFloatC(zero_allowed=False)
Positive0Float = PositiveFloatC(zero_allowed=True)
PositiveInt = click.IntRange(1,None)

def gauss1D(x,mean,var):
	return ((2 * pi) ** (-0.5)) * (1 / np.sqrt(var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)

# The program can run different tasks. Use the decorator @addtask to designate functions as tasks and to enable
# them to be run from the command line. Use the decorators @addparam and @addhyparam to register the parameters
# and hyperparameters to be added as @click.options. Here a hyperparameter is fixed during the program's
# execution whereas parameters are sampled for each run (randomly or in a grid), ie. they define the parameter 
# space.
#
# @addtask(task_name)
# @addhyparam(arg1, default_value=None, type=None)
# @addparam(arg2, default_lower_bound=None, default_upper_bound=None)
# def task_function(arg1, arg2, ..., **kwargs):
#	...
#
# All parameters given to the program will be passed to the corresponding function (as named parameters).
# It is best for the function to accept superfluous keyword arguments in case a parameter for another
# task is being passed to the program. This will avoid errors when calling the function in this case.
# However, parameters (not hyperparameters) not registered between @addtask and the function definition
# will not be saved in the output. In order to do this, the program keeps track of which parameters are
# registered by which task and filters them accordingly.
tasklist = {}
defaulttask = None
paramlist = []
hyparamlist = []

params_of_last_task = []

def addtask(name):
	global defaulttask
	defaulttask = name

	def addf(f):
		global params_of_last_task
		tasklist[name] = (f,params_of_last_task)
		params_of_last_task = []
		return f

	return addf

def addparam(name, defaultl, defaultu):
	global paramlist

	# Check for duplicate parameters
	add = True
	for p in paramlist:
		if p[0] == name:
			add = False
			params_of_last_task.append(p)

	if add:
		paramlist = paramlist + [(name, defaultl, defaultu)] 
		params_of_last_task.append(paramlist[-1])

	return lambda x: x

def addhyparam(name, default=None, ptype=None):
	global hyparamlist

	# Check for duplicate parameters and ensure they don't clash
	add = True
	for p in hyparamlist:
		if p[0] == name:
			add = False
			if p[2] != ptype:
				raise TypeError("Incompatible parameter types specified for hyperparameter '{}'".format(name))

	if add:
		hyparamlist = [(name, default, ptype)] + hyparamlist

	return lambda x: x

# SQRT MODEL
@addtask('sqrt')
@addparam('x', 0, np.sqrt(10))
@addhyparam('sigma', 0.05, PositiveFloat)
@addhyparam('eta', 3.0, PositiveFloat)
@addhyparam('mu', 5.0, float)
@addhyparam('kernel', 0.0, Positive0Float)
@addhyparam('x0', 1.333, float)
@addhyparam('nsamples', 1000000, PositiveInt)
def posterior(x, x0, kernel, sigma, eta, mu, nsamples, **kwargs):
	thetas = np.random.normal(mu, eta, size = nsamples)
	if kernel == 0:
		prep = gauss1D(x, np.sign(thetas) * (np.abs(thetas)) ** 0.3333, sigma) 
	else:
		prep = gauss1D(x,np.sqrt(np.abs(thetas)), sigma) * gauss1D(x,x0,kernel ** 2)
	post = prep / np.sum(prep)

	mean = np.sum(thetas * post)
	var = np.sum(((thetas - mean) ** 2) * post)
	return mean, var

# CALCULATE LOSS FUNCTION
def lqw(theta, x, alpha, beta, gamma, delta):
	return 0.5 * (np.log(2 * np.pi * (gamma ** 2)) + ((theta - alpha - beta * x - delta * (x ** 2)) / gamma) ** 2)

# See above for other parameters and hyperparameters
@addtask('lw')
@addparam('alpha', 0, 5)
@addparam('beta', 0, 5)
@addparam('gamma', 0, 5)
@addparam('delta', -2, 2)
def tasklw(alpha, x0, beta, gamma, delta, mu, sigma, eta, kernel, nsamples, **kwargs):
	thetas = np.random.normal(mu, eta, size = nsamples)
	xes = np.random.normal(np.sqrt(np.abs(thetas)), sigma)

	lqws = lqw(thetas, xes, alpha, beta, gamma, delta)
	if kernel == 0:
		lw = np.average(lqws)
	else:
		lw = np.average(lqws * gauss1D(xes,x0,kernel ** 2))
	return lw

# Find alternative file name to avoid overwriting existing files if this is specified
def findnewfilename(old):
	new = old

	for i in range(100):
		new = new + '_'	
		try:
			open(new, 'r')
		except IOError:
			return new

	raise IOError("Couldn't find alternative file to avoid overwriting '{}'".format(old))

# One-use decorator to add the previously supplied task parameters to click
def addtaskparams(func):
	ret = func
	for h in reversed(hyparamlist):
		currtype = float
		if len(h) >= 3:
			currtype = h[2]

		dec = click.option('--{}'.format(h[0]),type=currtype, default=h[1], show_default=True,help='Value for parameter \'{}\''.format(h[0]))
		ret = dec(ret)

	for p in reversed(paramlist):
		dec = click.option('--{}'.format(p[0]), type=(float,float), default=(p[1],p[2]), show_default=True,help='Range for parameter \'{}\''.format(p[0]))
		ret = dec(ret)

	return ret

def samplepts(n, num_cores, params, taskname, random = True):
	npertask = n // num_cores
	if n % num_cores != 0:
		npertask += 1

	draws = []

	if random == False:
		for ptype in paramlist:
			if ptype in tasklist[taskname][1]:
				p = params[ptype[0]]
				draws.append(np.linspace(p[0],p[1],num_cores * npertask).reshape((num_cores, npertask)))
	else:
		for ptype in paramlist:
			if ptype in tasklist[taskname][1]:
				p = params[ptype[0]]
				draws.append(np.random.uniform(p[0],p[1], (num_cores, npertask)))

	draws = np.transpose(draws, (1,2,0))
	return draws

def init_worker():
	signal.signal(signal.SIGINT, signal.SIG_IGN)

@click.command()
@click.argument('taskname', type=click.Choice(tasklist.keys()), default=defaulttask)
@click.argument('output', type=str)
@click.option('--num-cores', type=PositiveInt, default=4, help='Number of cores used')
@click.option('--nsims', type=PositiveInt, default=100, help='Number of points simulated')
@click.option('--random-samples/--no-random-samples', default=True, help='Sample parameters randomly instead of using a grid')
@click.option('--overwrite/--no-overwrite', default=False, help='Overwrite previously existing output files')
@addtaskparams
def main(output, taskname, num_cores, nsims, overwrite, random_samples, **params):
	pool = mp.Pool(num_cores, init_worker)
	hyparams = {}
	for k in params.keys():
		add = True
		for n in paramlist:
			if k == n[0]:
				add = False
				break

		if add:
			hyparams[k] = params[k]
	
	for k in hyparams.keys():
		del params[k]

	taskqueue = samplepts(nsims, num_cores, params, taskname, random_samples)
	argslist = [('{}{}'.format(output,i),taskqueue[i],hyparams,taskname) for i in range(num_cores)]

	try:
		rets = np.concatenate(pool.map(run, argslist))
	except KeyboardInterrupt:
		pool.terminate()
		pool.join()

	writeoutput(output, taskname, rets, hyparams, np.concatenate(taskqueue), overwrite)

# Output file format:
#
# Each process writes its output to a different file.
# The file name is determined by concatenating the value of the 'output' 
# argument with the ID of the process, from 0 to num_cores.
#
# If the file already exists, the new data will be appended.
# If the file already exists but is not readable, or if the hyperparameters differ,
# the program will write the output to a different file. No attempt at merging is made.
#
# The file contains a single python dictionary, whose entries are as follows:
#	
#	task:		Name of task
#	pts:		List of points in parameter space (see 'paramlist' for enumeration of parameters)
#	hyparams:		List of hyperparameters (see 'hyparamlist' for enumeration of hyperparameters) 
#	data:		Numerically evaluated Lw
#
def writeoutput(outname, taskname, results, hyparams, ptlist, overwrite):
	if not overwrite:
		try:
			inp = open(outname, 'rb')
			try:		
				indict = pickle.load(inp)
			except pickle.PicklingError:
				print("Error: Could not read from file '{}'".format(outname))
				outname = findnewfilename(outname)
				ret = 1
				print("Writing output to file '{}' instead".format(outname))
				raise IOError()

			if indict['hyparams'] != hyparams or indict['task'] != taskname:
				print("Warning: File '{}' contains data for different task (possibly different hyperparameters)".format(outname))
				outname = findnewfilename(outname)
				ret = 1
				print("Writing output to file '{}' instead".format(outname))
				raise IOError()


			ptlist = np.concatenate((indict['pts'],ptlist))
			results = np.concatenate((indict['data'],results))

		except IOError:
			pass

	outdict = { 'task' : taskname,'pts': ptlist, 'hyparams' : hyparams, 'data' : results}
	out = open(outname, 'wb')
	pickle.dump(outdict, out)
	out.close()
	
def run(args):
	ret = 0
	outname = args[0]
	ptlist = args[1]
	hyparams = args[2]
	taskname = args[3]
	task = tasklist[taskname][0]
	results = []
	for pt in ptlist:
		argdict = dict.copy(hyparams)
		j = 0
		for i in range(len(paramlist)):
			if paramlist[i] in tasklist[taskname][1]:
				argdict[paramlist[i][0]] = pt[j]
				j += 1
		results.append(task(**argdict))

	results = np.array(results)
	return results

if __name__ == '__main__':
	main()
