import numpy as np
cimport numpy as np
from libc.math cimport exp, log, cos, sqrt
from libc.stdlib cimport rand, srand, RAND_MAX
cimport cython

###############################################################################
# parameters
cdef double gbar_K = 13
cdef double gbar_Na = 10
cdef double g_L = 0.1
cdef double E_K = -90
cdef double E_Na = 50
cdef double E_L = -65
cdef double C = 1
cdef double Vt = -50.0
cdef double gbar_M = 0.1
cdef double tau_max = 1000

cdef double k_beta_n1 = 0.5
cdef double k_beta_n2 = 40
cdef double nois_fact = 0.05  # noise level

###############################################################################
# kinetics
@cython.cdivision(True)
cdef double alpha_n(double x):
	cdef double v1 = x - Vt - 15
	return -0.032*v1 / (exp(-0.2 * v1) - 1)

@cython.cdivision(True)
cdef double beta_n(double x):
	cdef double v1 = x - Vt - 10
	return k_beta_n1*exp(-v1/k_beta_n2)

@cython.cdivision(True)
cdef double alpha_m(double x):
	cdef double v1 = x - Vt - 13
	return -0.32*v1 / (exp(-0.25*v1) - 1)

@cython.cdivision(True)
cdef double beta_m(double x):
	cdef double v1 = x - Vt - 40
	return 0.28*v1 / (exp(0.2*v1)-1)

@cython.cdivision(True)
cdef double alpha_h(double x):
	cdef double v1 = x - Vt - 17
	return 0.128*exp(-v1/18)

@cython.cdivision(True)
cdef double beta_h(double x):
	cdef double v1 = x - Vt - 40
	return 4.0/(1 + exp(-0.2*v1))

@cython.cdivision(True)
cdef double p_inf(double x):
	cdef double v1 = x + 35
	return 1.0/(1 + exp(-0.1*v1))

@cython.cdivision(True)
cdef double tau_p(double x):
	cdef double v1 = x + 35
	return tau_max/(3.3*exp(0.05*v1) + exp(-0.05*v1))

@cython.cdivision(True)
cdef double normal():
	cdef double u1 = rand() * 1.0 / RAND_MAX
	cdef double u2 = rand() * 1.0 / RAND_MAX

	return sqrt(-2 * log(u1)) * cos(2 * 3.141592658539 * u2)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void updatefe(np.ndarray[double,ndim=1] I, np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,int i,double tstep):
	# currents
	cdef double I_Na = (m[i-1]**3)*gbar_Na*h[i-1]*(V[i-1]-E_Na)
	cdef double I_K = (n[i-1]**4)*gbar_K*(V[i-1]-E_K)
	cdef double I_L = g_L*(V[i-1]-E_L)
	cdef double I_M = gbar_M*p[i-1]*(V[i-1]-E_K)
	cdef double I_ion = I[i-1] - I_K - I_Na - I_L - I_M

	###############################
	# + nois_fact*np.random.normal(1)/(tstep**0.5)
	V[i] = V[i-1] + tstep*(I_ion + nois_fact*normal()/(tstep**0.5))/C
	cdef double Vint = V[i-1]
	n[i] = n[i-1] + tstep*(alpha_n(Vint)*(1-n[i-1]) - beta_n(Vint)*n[i-1])
	m[i] = m[i-1] + tstep*(alpha_m(Vint)*(1-m[i-1]) - beta_m(Vint)*m[i-1])
	h[i] = h[i-1] + tstep*(alpha_h(Vint)*(1-h[i-1]) - beta_h(Vint)*h[i-1])
	p[i] = p[i-1] + tstep*(p_inf(Vint)-p[i-1])/tau_p(Vint)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void updatelf(np.ndarray[double,ndim=1] I, np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,int i,double tstep):
	# currents
	cdef double I_Na = (m[i-1]**3)*gbar_Na*h[i-1]*(V[i-1]-E_Na)
	cdef double I_K = (n[i-1]**4)*gbar_K*(V[i-1]-E_K)
	cdef double I_L = g_L*(V[i-1]-E_L)
	cdef double I_M = gbar_M*p[i-1]*(V[i-1]-E_K)
	cdef double I_ion = I[i-1] - I_K - I_Na - I_L - I_M

	###############################
	# + nois_fact*np.random.normal(1)/(tstep**0.5)
	V[i] = V[i-1] + tstep*(I_ion + nois_fact*normal()/(tstep**0.5))/C
	cdef double Vint = V[i]
	n[i] = n[i-1] + tstep*(alpha_n(Vint)*(1-n[i-1]) - beta_n(Vint)*n[i-1])
	m[i] = m[i-1] + tstep*(alpha_m(Vint)*(1-m[i-1]) - beta_m(Vint)*m[i-1])
	h[i] = h[i-1] + tstep*(alpha_h(Vint)*(1-h[i-1]) - beta_h(Vint)*h[i-1])
	p[i] = p[i-1] + tstep*(p_inf(Vint)-p[i-1])/tau_p(Vint)

# Leapfrog
def computelf(np.ndarray[double,ndim=1] t,np.ndarray[double,ndim=1] I,np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,double tstep,int seed=100):
	srand(seed)
	V[0] = E_L
	n[0] = alpha_n(V[0])/(alpha_n(V[0])+beta_n(V[0]))
	m[0] = alpha_m(V[0])/(alpha_m(V[0])+beta_m(V[0]))
	h[0] = alpha_h(V[0])/(alpha_h(V[0])+beta_h(V[0]))
	p[0] = p_inf(V[0])

	for i in range(1, t.shape[0]):
		updatelf(I,V,m,n,h,p,i,tstep)

# Forward Euler
def computefe(np.ndarray[double,ndim=1] t,np.ndarray[double,ndim=1] I,np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,double tstep,int seed=100):
	srand(seed)
	V[0] = E_L
	n[0] = alpha_n(V[0])/(alpha_n(V[0])+beta_n(V[0]))
	m[0] = alpha_m(V[0])/(alpha_m(V[0])+beta_m(V[0]))
	h[0] = alpha_h(V[0])/(alpha_h(V[0])+beta_h(V[0]))
	p[0] = p_inf(V[0])

	for i in range(1, t.shape[0]):
		updatefe(I,V,m,n,h,p,i,tstep)

def setparams(params):
	global gbar_Na, gbar_K, g_L, E_Na, E_K, E_L, gbar_M, tau_max, k_beta_n1, k_beta_n2, Vt, noise_fact

	gbar_Na = params[0,0]
	gbar_K = params[0,1]
	g_L = params[0,2]
	E_Na = params[0,3]
	E_K = -params[0,4]
	E_L = params[0,5]
	gbar_M = params[0,6]
	tau_max = params[0,7]
	k_beta_n1 = params[0,8]
	k_beta_n2 = params[0,9]
	Vt = -params[0,10]
	noise_fact = params[0,11]
