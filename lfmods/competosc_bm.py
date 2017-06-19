import numpy as np

class CO:
    def __init__(self, init, params, seed=None):
        self.state = np.asarray(init)
        self.params = np.asarray(params)

        # note: make sure to generate all randomness through self.rng (!)
        self.seed = seed
        if seed is not None:
            self.rng = np.random.RandomState(seed=seed)
        else:
            self.rng = np.random.RandomState()

    def sim_time(self, dt, t, max_n_steps=float('inf')):
        """Simulates the model."""

        # parameters
        gsynA = self.params[0,0]  # nS
        gsynA.astype(float)
        gelec = self.params[0,1]
        gelec.astype(float)
        tstep = float(dt)

        # fixed parameter
        ghc = 0.005

        ########################################################################
        # simulator
        def switchIC(t,y):
            ny = len(y)
            # initials conditions
            N = y[0:5]
            H = y[5:10]
            Vm = y[10:15]

            gsyn_lp_pd = ghc #uS
            gsyn_lg_in = ghc
            gsyn_ic_pd = gsynA
            gsyn_ic_in = gsynA
            gel_ic_lp = gelec
            gel_ic_lg = gelec

            # ionic conductances (f1, f2, hn, s2, s1)
            gk = np.array([0.039,0.039,0.019,0.015,0.015])
            gl = np.array([0.0001,0.0001,0.0001,0.0001,0.0001])
            gc = np.array([0.019,0.019,0.017,0.0085,0.0085])
            gh = np.array([0.025,0.025,0.008,0.01,0.01])

            n = 5
            iext=np.zeros(n)

            c=1 #nF
            phi=0.002 #1/ms
            vk=-80 #mV
            vl=-40
            vca=100
            vh=-20
            vsyn=-75
            vp1=0
            vp2=20
            vp3=0
            vp4=15
            vp5=78.3
            vp6=10.5
            vp7=-42.2
            vp8=87.3
            vth=-25
            vp11=5

            minf=.5*(1+np.tanh((Vm-vp1)/vp2))
            ninf=.5*(1+np.tanh((Vm-vp3)/vp4))
            lamdn= phi*np.cosh((Vm-vp3)/(2*vp4))
            hinf=1/(1+np.exp((Vm+vp5)/vp6))
            tauh=(272-((-1499)/(1+np.exp((-Vm+vp7)/vp8))))
            # syn from cell onto others
            sinf=1/(1+np.exp((vth-Vm)/vp11))

            ielec = np.array([0,
                              gel_ic_lp*(Vm[1]-Vm[2]),
                              (gel_ic_lp*(Vm[2]-Vm[1]))+(gel_ic_lg*(Vm[2]-Vm[3])),
                              gel_ic_lg*(Vm[3]-Vm[2]),
                              0])

            isyn = np.array([(gsyn_lp_pd*sinf[1]*(Vm[0]-vsyn)),
                             (gsyn_lp_pd*sinf[0]*(Vm[1]-vsyn)),
                             (gsyn_ic_pd*sinf[0]*(Vm[2]-vsyn))+(gsyn_ic_in*sinf[4]*(Vm[2]-vsyn)),
                             (gsyn_lg_in*sinf[4]*(Vm[3]-vsyn)),
                             (gsyn_lg_in*sinf[3]*(Vm[4]-vsyn))])

            ica = gc*minf*(Vm-vca)
            ik = gk*N*(Vm-vk)
            ih = gh*H*(Vm-vh)
            il = gl*(Vm-vl) #nA

            dy = np.zeros(ny)    # a column vector

            dy[0:5] = lamdn*(ninf-N) #dN
            dy[5:10] = (hinf-H)/tauh #dH
            dy[10:15] = (iext-ica-il-ik-ih-ielec-isyn)/c

            return dy

        ########################################################################
        # simulation

        # network size
        n = 5

        # initial conditions
        N0 = np.zeros((1,n))
        H0 = np.zeros((1,n))
        randinit = 0.1*(self.rng.rand(1,n)-0.5) #+/- 0.05
        Vm0 = -65.+randinit
        y0 = np.concatenate((N0,H0,Vm0),axis=1)[0,:]

        # store trajectories
        Y = np.zeros((t.shape[0], n*3))
        Y[0,:] = y0


        for i in range(1, t.shape[0]):
            Y[i,:] = Y[i-1,:] + dt*switchIC(t[i-1],Y[i-1,:])

        V = Y[:,13]

        return np.array(V).reshape(-1,1)
