import numpy as np
cimport numpy as np
import scipy
from libc.math cimport exp, log, cos, sqrt
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
cdef double nois_fact = 0.5

###############################################################################
# kinetics
@cython.cdivision(True)
cdef double alpha_n(double x):
	cdef double v1 = x - Vt - 15
	return -0.032*v1 / (exp(-0.2 * v1) - 1)
	
@cython.cdivision(True)
cdef double dalpha_n(double x):
	cdef double v1 = x - Vt - 15
	cdef double den = exp(-0.2 * v1) - 1
	return -0.032*(den + v1 * 0.2 * (den + 1)) / (den ** 2)

@cython.cdivision(True)
cdef double beta_n(double x):
	cdef double v1 = x - Vt - 10
	return k_beta_n1*exp(-v1/k_beta_n2)
	
@cython.cdivision(True)
cdef double dbeta_n(double x):
	cdef double v1 = x - Vt - 10
	return -k_beta_n1 / k_beta_n2 * exp(-v1 / k_beta_n2)

@cython.cdivision(True)
cdef double alpha_m(double x):
	cdef double v1 = x - Vt - 13
	return -0.32*v1 / (exp(-0.25*v1) - 1)
	
@cython.cdivision(True)
cdef double dalpha_m(double x):
	cdef double v1 = x - Vt - 13
	cdef double den = exp(-0.25 * v1) - 1
	return -0.32*(den + v1 * 0.25 * (den + 1)) / (den ** 2)

@cython.cdivision(True)
cdef double beta_m(double x):
	cdef double v1 = x - Vt - 40
	return 0.28*v1 / (exp(0.2*v1)-1)

@cython.cdivision(True)
cdef double dbeta_m(double x):
	cdef double v1 = x - Vt - 40
	cdef double den = exp(0.2 * v1) - 1
	return 0.28*(den - v1 * 0.2 * (den + 1)) / (den ** 2)

@cython.cdivision(True)
cdef double alpha_h(double x):
	cdef double v1 = x - Vt - 17
	return 0.128*exp(-v1/18)

@cython.cdivision(True)
cdef double dalpha_h(double x):
	cdef double v1 = x - Vt - 17
	return -0.128*exp(-v1/18) / 18

@cython.cdivision(True)
cdef double beta_h(double x):
	cdef double v1 = x - Vt - 40
	return 4.0/(1 + exp(-0.2*v1))

@cython.cdivision(True)
cdef double dbeta_h(double x):
	cdef double v1 = x - Vt - 40
	cdef double den = exp(-0.2 * v1) + 1
	return 4.0 * (0.2 * (den - 1)) / (den ** 2)

@cython.cdivision(True)
cdef double p_inf(double x):
	cdef double v1 = x + 35
	return 1.0/(1 + exp(-0.1*v1))

@cython.cdivision(True)
cdef double dp_inf(double x):
	cdef double v1 = x + 35
	cdef double den = exp(-0.1 * v1) + 1
	return 1.0 * (0.1 * (den - 1)) / (den ** 2)

@cython.cdivision(True)
cdef double tau_p(double x):
	cdef double v1 = x + 35
	return tau_max/(3.3*exp(0.05*v1) + exp(-0.05*v1))
	
@cython.cdivision(True)
cdef double dtau_p(double x):
	cdef double v1 = x + 35
	cdef double den = 3.3*exp(0.05*v1) + exp(-0.05*v1)
	return -tau_max * (0.05 * 3.3 * exp(0.05 * v1) - 0.05 * exp(-0.05*v1)) / (den ** 2)

def seed(n):
	np.random.seed(n)

def setnoisefactor(double x):
	global nois_fact
	nois_fact = x

def setparams(params):
	global gbar_Na, gbar_K, g_L, E_Na, E_K, E_L, gbar_M, tau_max, k_beta_n1, k_beta_n2, Vt, nois_fact

	gbar_Na = params[0,0]
	gbar_K = params[0,1]
	g_L = params[0,2]
	E_Na = params[0,3]
	E_K = -params[0,4]
	E_L = -params[0,5]
	gbar_M = params[0,6]
	tau_max = params[0,7]
	k_beta_n1 = params[0,8]
	k_beta_n2 = params[0,9]
	Vt = -params[0,10]
	nois_fact = params[0,11]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void updatebe(np.ndarray[double,ndim=1] I, np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,int i,double tstep,int fineness):
	cdef np.ndarray[double,ndim=1] b = np.array([V[i-1],n[i-1],m[i-1],h[i-1],p[i-1]])
	cdef double dt = tstep / fineness

	for j in range(fineness):
		b = scipy.optimize.fixed_point(updatefunc,b,args=(b,I[i-1],dt))

	V[i] = b[0] + nois_fact * (np.random.normal() * sqrt(dt))
	n[i] = b[1]
	m[i] = b[2]
	h[i] = b[3]
	p[i] = b[4]

# Backward Euler
def backwardeuler(np.ndarray[double,ndim=1] t,np.ndarray[double,ndim=1] I,np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,double tstep,int fineness):
	V[0] = E_L
	n[0] = alpha_n(V[0])/(alpha_n(V[0])+beta_n(V[0]))
	m[0] = alpha_m(V[0])/(alpha_m(V[0])+beta_m(V[0]))
	h[0] = alpha_h(V[0])/(alpha_h(V[0])+beta_h(V[0]))
	p[0] = p_inf(V[0])

	for i in range(1, t.shape[0]):
		updatebe(I,V,m,n,h,p,i,tstep,fineness)

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
def updatefunc(np.ndarray[double,ndim=1] a,np.ndarray[double,ndim=1] b,double I,double tstep):
	cdef np.ndarray[double,ndim=1] ret = np.empty(5)

	# currents
	cdef double I_Na = (a[2]**3)*gbar_Na*a[3]*(a[0]-E_Na)
	cdef double I_K = (a[1]**4)*gbar_K*(a[0]-E_K)
	cdef double I_L = g_L*(a[0]-E_L)
	cdef double I_M = gbar_M*a[4]*(a[0]-E_K)
	cdef double I_ion = I - I_K - I_Na - I_L - I_M
	
	###############################
	ret[0] = b[0] + tstep*(I_ion)/C
	ret[1] = b[1] + tstep*(alpha_n(a[0])*(1-a[1]) - beta_n(a[0])*a[1])
	ret[2] = b[2] + tstep*(alpha_m(a[0])*(1-a[2]) - beta_m(a[0])*a[2])
	ret[3] = b[3] + tstep*(alpha_h(a[0])*(1-a[3]) - beta_h(a[0])*a[3])
	ret[4] = b[4] + tstep*(p_inf(a[0])-a[4])/tau_p(a[0])

	return ret

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void updatefe(np.ndarray[double,ndim=1] I, np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,int i,double tstep, int fineness):
	# currents
	cdef double cV = V[i-1]
	cdef double cm = m[i-1]
	cdef double cn = n[i-1]
	cdef double ch = h[i-1]
	cdef double cp = p[i-1]

	cdef double I_Na, I_K, I_L, I_M, I_ion
	cdef double dt = tstep / fineness

	cdef int j
	cdef np.ndarray[double,ndim=1] rl = np.random.normal(size = fineness)

	for j in range(fineness):
		I_Na = (cm**3)*gbar_Na*ch*(cV-E_Na)
		I_K = (cn**4)*gbar_K*(cV-E_K)
		I_L = g_L*(cV-E_L)
		I_M = gbar_M*cp*(cV-E_K)
		I_ion = I[i-1] - I_K - I_Na - I_L - I_M
	
		###############################
		cV += (dt*I_ion + nois_fact * (rl[j] * sqrt(dt))) / C
		cn += dt*(alpha_n(cV)*(1-cn) - beta_n(cV) * cn)
		cm += dt*(alpha_m(cV)*(1-cm) - beta_m(cV) * cm)
		ch += dt*(alpha_h(cV)*(1-ch) - beta_h(cV) * ch)
		cp += dt*(p_inf(cV)-cp)/tau_p(cV)

	V[i] = cV
	n[i] = cn
	m[i] = cm
	h[i] = ch
	p[i] = cp

# Forward Euler
def forwardeuler(np.ndarray[double,ndim=1] t,np.ndarray[double,ndim=1] I,np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p, double tstep, int fineness):
	V[0] = E_L
	n[0] = alpha_n(V[0])/(alpha_n(V[0])+beta_n(V[0]))
	m[0] = alpha_m(V[0])/(alpha_m(V[0])+beta_m(V[0]))
	h[0] = alpha_h(V[0])/(alpha_h(V[0])+beta_h(V[0]))
	p[0] = p_inf(V[0])

	for i in range(1, t.shape[0]):
		updatefe(I,V,m,n,h,p,i,tstep,fineness)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
@cython.cdivision(True)
cdef void updatehines(np.ndarray[double,ndim=1] I, np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,int i,double tstep,int fineness):
	# currents
	cdef double cV = V[i-1]
	cdef double cm = m[i-1]
	cdef double cn = n[i-1]
	cdef double ch = h[i-1]
	cdef double cp = p[i-1]

	cdef double I_Na, I_K, I_L, I_M, I_ion
	cdef double dt = tstep / fineness

	cdef int j
	
	cdef double A
	cdef double B
	cdef double Cn, Dn, Cm, Dm, Ch, Dh, Cp, Dp

	cdef double dw1, dw2
	cdef double dz1, dz2
	cdef double q

	cdef np.ndarray[double,ndim=1] rl = np.random.normal(size = 2 * fineness)
	cdef np.ndarray[double,ndim=1] wl = np.random.normal(size = 2 * fineness)
	for j in range(fineness):
		A = -(cm ** 3)*gbar_Na*ch - (cn ** 4) * gbar_K - g_L - gbar_M * cp
		B = I[i-1] + (cm ** 3) * gbar_Na * ch * E_Na + (cn**4) * gbar_K * E_K + g_L * E_L + gbar_M * cp * E_K
	
		dw1 = wl[2 * j] * sqrt(0.5 * dt)
		dw2 = wl[2 * j + 1] * sqrt(0.5 * dt)
		dz1 = 0.5 * (dt / 2.0) * (dw1 + rl[2 * j] * sqrt(dt / 3.0))
		dz2 = 0.5 * (dt / 2.0) * (dw2 + rl[2 * j + 1] * sqrt(dt / 3.0))

		q = dz1 + dz2 - 0.5 * dt * dw1

		cV = (cV + dt * (A * 0.5 * (cV + dw1 - dw2) + B) + nois_fact * (dw1 + dw2) + nois_fact * A * q) / (1.0 - 0.5 * dt * A)

		Cn = alpha_n(cV)
		Cm = alpha_m(cV)
		Ch = alpha_h(cV)
		Cp = p_inf(cV) / tau_p(cV)

		Dn = -(Cn + beta_n(cV))
		Dm = -(Cm + beta_m(cV))
		Dh = -(Ch + beta_h(cV))
		Dp = -1.0 / tau_p(cV)

		cn = (cn + dt * (0.5 * Dn * cn + Cn)) / (1.0 - 0.5 * dt * Dn)
		cm = (cm + dt * (0.5 * Dm * cm + Cm)) / (1.0 - 0.5 * dt * Dm)
		ch = (ch + dt * (0.5 * Dh * ch + Ch)) / (1.0 - 0.5 * dt * Dh)
		cp = (cp + dt * (0.5 * Dp * cp + Cp)) / (1.0 - 0.5 * dt * Dp)

	V[i] = cV
	n[i] = cn
	m[i] = cm
	h[i] = ch
	p[i] = cp

# Leapfrog
def hinesmethod(np.ndarray[double,ndim=1] t,np.ndarray[double,ndim=1] I,np.ndarray[double,ndim=1] V,np.ndarray[double,ndim=1] m,np.ndarray[double,ndim=1] n,np.ndarray[double,ndim=1] h,np.ndarray[double,ndim=1] p,double tstep,int fineness):
	V[0] = E_L
	n[0] = alpha_n(V[0])/(alpha_n(V[0])+beta_n(V[0]))
	m[0] = alpha_m(V[0])/(alpha_m(V[0])+beta_m(V[0]))
	h[0] = alpha_h(V[0])/(alpha_h(V[0])+beta_h(V[0]))
	p[0] = p_inf(V[0])

	for i in range(1, t.shape[0] - 1):
		updatehines(I,V,m,n,h,p,i,tstep,fineness)

	V[t.shape[0] - 1] = V[t.shape[0] - 2]
