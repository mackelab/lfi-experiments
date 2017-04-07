import numpy as np

class HH:
    def __init__(self, init, params, seed=None):
        self.state = np.asarray(init)
        self.params = np.asarray(params)

        # note: make sure to generate all randomness through self.rng (!)
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    def sim_time(self, dt, t, I, max_n_steps=float('inf')):
        """Simulates the model for a specified time duration."""

        gbar_Na = self.params[0,0]  # mS/cm2
        gbar_Na.astype(float)
        gbar_K = self.params[0,1]
        gbar_K.astype(float)
        g_leak = self.params[0,2]
        g_leak.astype(float)
        E_Na = self.params[0,3]  # mV
        E_Na.astype(float)
        E_K = -self.params[0,4]
        E_K.astype(float)
        E_leak = -self.params[0,5]
        E_leak.astype(float)
        gbar_M = self.params[0,6]
        gbar_M.astype(float)
        tau_max = self.params[0,7]  # ms
        tau_max.astype(float)
        k_beta_n1 = self.params[0,8]
        k_beta_n1.astype(float)
        k_beta_n2 = self.params[0,9]
        k_beta_n2.astype(float)
        Vt = -self.params[0,10]
        Vt.astype(float)
        nois_fact = self.params[0,11]
        nois_fact.astype(float)
        tstep = float(dt)

        # Parameters

        # gbar_Na = 50
        # gbar_K = 5
        # g_leak = 0.1
        # E_Na = 50.
        # E_K = -90.
        # E_leak = -70.
        # gbar_M = 0.07
        # tau_max = 600.
        # k_beta_n1 = 0.5
        # k_beta_n2 = 40.
        # Vt = -60.0

        # nois_fact = 0.
        nois_fact_obs = 0.
        C = 1.  # uF/cm2

        # to generate burst
        # gbar_L = 0.1
        # E_Ca = 120
        # gbar_T = 0.4
        # Vx = 2

        ####################################

        # kinetics

        # voltage dependent Na
        def alpha_m(x):
            v1 = x - Vt - 13.
            return -0.32*v1 / (np.exp(-0.25*v1) - 1.)

        def beta_m(x):
            v1 = x - Vt - 40
            return 0.28*v1 / (np.exp(0.2*v1)-1.)

        def alpha_h(x):
            v1 = x - Vt - 17.
            return 0.128*np.exp(-v1/18.)

        def beta_h(x):
            v1 = x - Vt - 40.
            return 4.0/(1 + np.exp(-0.2*v1))

        # delayed-rectifier K+
        def alpha_n(x):
            v1 = x - Vt - 15.
            return -0.032*v1 / (np.exp(-0.2 * v1) - 1)

        def beta_n(x):
            v1 = x - Vt - 10.
            return k_beta_n1*np.exp(-v1/k_beta_n2)

        # slow non-inactivating K+
        def p_inf(x):
            v1 = x + 35.
            return 1.0/(1. + np.exp(-0.1*v1))

        def tau_p(x):
            v1 = x + 35.
            return tau_max/(3.3*np.exp(0.05*v1) + np.exp(-0.05*v1))

#        # to generate burst
#        # high-threshold Ca2+
#        def alpha_q(x):
#            v1 = x + 27
#            return 0.055*v1 / (1 - np.exp(-v1/3.8))
#
#        def beta_q(x):
#            v1 = x + 75
#            return 0.94*np.exp(-v1/17)
#
#        def alpha_r(x):
#            v1 = x + 13
#            return 0.000457*np.exp(-v1/50)
#
#        def beta_r(x):
#            v1 = x + 15
#            return 0.0065/(1 + np.exp(-v1/28))
#
#        # low-threshold Ca2+
#        def s_inf(x):
#            v1 = x + Vx + 57
#            return 1/(1 + np.exp(-v1/6.2))
#
#        def u_inf(x):
#            v1 = x + Vx + 81
#            return 1/(1 + np.exp(v1/4))
#
#        def tau_u(x):
#            v1 = x + Vx + 84
#            v2 = x + Vx + 113.2
#            return 30.8/3.7 + (211.4 + np.exp(v2/5))/(3.7*(1 + np.exp(v1/3.2)))


        ####################################

        # simulation from initial point
        V = np.zeros_like(t) # baseline voltage
        n = np.zeros_like(t)
        m = np.zeros_like(t)
        h = np.zeros_like(t)
        p = np.zeros_like(t)
#        q = np.zeros_like(t)
#        r = np.zeros_like(t)
#        u = np.zeros_like(t)

#        V[0] = float(self.state[0])
        V[0] = E_leak
        n[0] = alpha_n(V[0])/(alpha_n(V[0])+beta_n(V[0]))
        m[0] = alpha_m(V[0])/(alpha_m(V[0])+beta_m(V[0]))
        h[0] = alpha_h(V[0])/(alpha_h(V[0])+beta_h(V[0]))
        p[0] = p_inf(V[0])
#        q[0] = alpha_q(V[0])/(alpha_q(V[0])+beta_q(V[0]))
#        r[0] = alpha_r(V[0])/(alpha_r(V[0])+beta_r(V[0]))
#        u[0] = u_inf(V[0])

        for i in range(1, t.shape[0]):

            # currents
            I_Na = (m[i-1]**3)*gbar_Na*h[i-1]*(V[i-1]-E_Na)
            I_K = (n[i-1]**4)*gbar_K*(V[i-1]-E_K)
            I_leak = g_leak*(V[i-1]-E_leak)
            I_M = gbar_M*p[i-1]*(V[i-1]-E_K)
#            I_L = (q[i-1]**2)*gbar_L*r[i-1]*(V[i-1]-E_Ca)
#            I_T = (s_inf(V[i-1])**2)*u[i-1]*gbar_T*(V[i-1]-E_Ca)
            I_ion = I[i-1] - I_K - I_Na - I_leak - I_M

            ###############################
            V[i] = V[i-1] + tstep*(I_ion + nois_fact*self.rng.randn()/(tstep**0.5))/C
            n[i] = n[i-1] + tstep*(alpha_n(V[i-1])*(1-n[i-1]) - beta_n(V[i-1])*n[i-1])
            m[i] = m[i-1] + tstep*(alpha_m(V[i-1])*(1-m[i-1]) - beta_m(V[i-1])*m[i-1])
            h[i] = h[i-1] + tstep*(alpha_h(V[i-1])*(1-h[i-1]) - beta_h(V[i-1])*h[i-1])
            p[i] = p[i-1] + tstep*(p_inf(V[i-1])-p[i-1])/tau_p(V[i-1])
#            q[i] = q[i-1] + tstep*(alpha_q(V[i-1])*(1-q[i-1]) - beta_q(V[i-1])*q[i-1])
#            r[i] = r[i-1] + tstep*(alpha_r(V[i-1])*(1-r[i-1]) - beta_r(V[i-1])*r[i-1])
#            u[i] = u[i-1] + tstep*(u_inf(V[i-1])-u[i-1])/tau_u(V[i-1])

#        return np.array(V).reshape(-1,1)
        return np.array(V).reshape(-1,1) + nois_fact_obs*self.rng.randn(t.shape[0],1)
