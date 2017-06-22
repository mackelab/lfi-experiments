import sys
import numpy as np
import likelihoodfree.io as io
import subprocess as sp
import matplotlib.pyplot as plt

runs = []
outdirname = 'benchmark'
problem = 'gauss'

class KernelParam:
	def __init__(self, name = None, bw = None):
		self.name = name
		self.bw = bw

	def getargs(self):
		if self.name == None:
			return []

		return [ '--loss-calib', '{}'.format(self.bw) , '--loss-calib-kernel', '{}'.format(self.name) ]

def calcdiff(p1, p2):
	xlist = np.linspace(-10,10,500).reshape((-1,1))
	p1list = p1.eval(xlist,log=False)
	p2list = p2.eval(xlist,log=False)
	return np.sum(np.abs(p1list - p2list))

def plotpdfs(*pdfs):
	fig, ax = plt.subplots(1)
	xlist = np.linspace(-10,10,500)
	xlist_ = xlist.reshape((-1,1))
	ylists = [ pdf.eval(xlist_,log=False) for pdf in pdfs ]
	for i in range(len(ylists)):
		ax.plot(xlist,ylists[i])

	plt.draw()

def runsim(kernelparam, output_filename):
	python_path = 'python3'
	execname = './run.py'
	nreps = 15
	nsamples = 50

	output_path = '{}/{}'.format(outdirname, output_filename)
	data = { 'kernel': kernelparam, 'out' : output_filename }
	runs.append(data)
	extra_args = [ '--samples', '{}'.format(nsamples) ] + kernelparam.getargs()
	try:
		output = sp.check_output([python_path, execname, problem, output_path, '--iw-loss', '--rep', '{}'.format(nreps)] + extra_args)
	except sp.CalledProcessError as err:
		print("An error occurred while running '{}'.".format(execname))
		sys.exit(1)

def runsims():
	kp = KernelParam()
	output_filename = 'nokernel'
	runsim(kp, output_filename)

	kernels = [ KernelParam('gauss', 10 ** (0.25 * i)) for i in range(0,-12,-1) ]
	for kernel in kernels:
		output_filename = '{}_bw{}'.format(kernel.name, kernel.bw)
		runsim(kernel, output_filename)

def testresults():
	dirpath = 'results/{}/nets/{}/'.format(problem, outdirname)
	for run in runs:
		dists, infos, losses, nets, posteriors, sims = io.load_prefix(dirpath, run['out'])

		sim = io.first(sims)
		xobs = np.asarray([[0]])

		print("Considering run '{}'".format(io.first(infos)["prefix"]))
		print("Kernel bandwidth: {}".format(run["kernel"].bw))
		print("Observed value: {}".format(xobs))

		errs = []

		for i in range(len(nets)):
			print("Iteration #{}".format(i))
			net = io.nth(nets, i)
			sim = io.nth(sims, i)
			posterior = io.nth(posteriors, i)

			true_posterior = sim.posterior
			
# 			if hasattr(true_posterior,'xs'):
# 				print(len(true_posterior.xs))
# 				print(true_posterior.xs[0])
# 				assert(len(true_posterior.xs) == 1)
# 				true_mean = true_posterior.xs[0].m
# 				true_sd = true_posterior.xs[0].S
# 			else:
# 				true_mean = sim.posterior.m
# 				true_sd = sim.posterior.S
# 
# 			assert(len(posterior.xs) == 1)
# 			post_mean = posterior.xs[0].m
# 			post_sd = posterior.xs[0].S
	
# 			plotpdfs(posterior, true_posterior)
			err = calcdiff(posterior, true_posterior)
			print("L1 difference in pdfs:")
			print('{}'.format(err))

			errs.append(err)

		fig, axm = plt.subplots(1, sharex=True)
		xlist = range(len(nets))
		fig.suptitle("Error in '{}'".format(run["out"]))
		axm.plot(xlist, errs)
		axm.set_ylabel("L1 Error")
		axm.xaxis.set_ticks(xlist)
		#axm.set_xlabel("Runs")
		axm.set_xlabel("Runs")
		plt.draw()

	plt.show()

if __name__ == '__main__':
	runsims()
	testresults()
