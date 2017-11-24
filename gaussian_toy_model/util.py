import delfi.distribution as dd
import delfi.generator as dg
import delfi.inference as infer
import delfi.summarystats as ds
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import multivariate_normal as mvn
from scipy.special import gammaln
from delfi.utils.progress import no_tqdm, progressbar

from delfi.simulator.BaseSimulator import BaseSimulator
from delfi.simulator.Gauss import Gauss
    
def gauss_weights(params, stats, mu_y, Sig_y):
    
    y = np.hstack((stats, params))
    return mvn.pdf(x=y, mean=mu_y.reshape(-1), cov=Sig_y, allow_singular=True)   


def gauss_weights_eps0(params, stats, mu_y, Sig_y):
    """ stable version in case eps^2 is giant - stats.mvn return nonsense here """
    # Note: making use of the fact that covariances are zero for normal weights in SNPE/CDELFI MLE solutions
    
    x = -0.5 * (params-mu_y[1])**2 / Sig_y[1,1] # would like to use mvn.pdf, but that one freaks  
    return np.exp( x.reshape(-1) ) # out for 1D problems with negative (co-)variance


def sel_gauss_implementation(eps2, thresh=1000): 
    
    return gauss_weights_eps0 if eps2 > thresh else gauss_weights


#def studentT_weights(params, stats, mu_y, Sig_y):
#    
#    raise NotImplementedError
#    
#    
#def studentT_weights_eps0(params, stats, mu_y, Sig_y, df=3):
#    """ stable version in case eps^2 is giant - stats.mvn return nonsense here """
#    # Note: making use of the fact that covariances are zero for normal weights in SNPE/CDELFI MLE solutions
#    
#    exponent = -(df+1)/2 
#    return (1 + (params-mu_y[1])**2/(df*Sig_y[1,1]))**exponent
#
#
#def sel_studentT_implementation(eps2, thresh=1000): 
#    
#    return studentT_weights_eps0 if eps2 > thresh else studentT_weights
#

def get_weights_fun(eps2, thresh=1000, proposal_form='normal'):
    
    if proposal_form=='normal':
        selector = sel_gauss_implementation 
    elif proposal_form=='studentT':
        selector = sel_studentT_implementation 
        
    return selector(eps2, thresh)

def get_weights(proposal_form, eta2, ksi2, eps2, x0, nu, stats, params, df=3, a=None, b=None): 
    
    assert proposal_form in ('normal', 'studentT', 'unif')
    
    if proposal_form == 'normal':
        
        proposal_pdf = mvn.pdf(x=params, mean=nu, cov=ksi2)
        prior_pdf    = mvn.pdf(x=params, mean=0., cov=eta2)
        normals = prior_pdf / proposal_pdf
        if eps2 < 1000:
            calibration_kernel_pdf = mvn.pdf(x=stats, mean=x0, cov=eps2)
            normals *= calibration_kernel_pdf
        
    if proposal_form == 'studentT':

        exponent = -(df+1)/2 
        proposal_pdf = (1 + (params-nu)**2/(df*ksi2))**exponent
        proposal_pdf *= np.exp(gammaln((df + 1) / 2)-gammaln(df / 2) - \
            1/2 * np.log(df * np.pi) - 0.5 * np.log(ksi2)) 
        prior_pdf    = mvn.pdf(x=params, mean=0., cov=eta2)
        normals = prior_pdf / proposal_pdf
        if eps2 < 1000:
            calibration_kernel_pdf = mvn.pdf(x=stats, mean=x0, cov=eps2)
            normals *= calibration_kernel_pdf
            
    if proposal_form == 'unif':
        
        proposal_pdf = (1./(b-a)) * np.ones_like(params) / (1.-((params > b) + (params < a)))
        prior_pdf    = mvn.pdf(x=params, mean=0., cov=eta2)
        normals = prior_pdf / proposal_pdf
        if eps2 < 1000:
            calibration_kernel_pdf = mvn.pdf(x=stats, mean=x0, cov=eps2)
            normals *= calibration_kernel_pdf
            
            
    return normals



def dL_dalpha( params, stats, normals, beta, gamma2, alphas):

    return -2*(normals.reshape(-1,1) * (params.reshape(-1,1) - beta * stats.reshape(-1,1) - alphas.reshape(1,-1))/gamma2 * stats.reshape(-1,1)).sum(axis=0)


def dL_dbeta( params, stats, normals, alpha, gamma2, betas):

    return -2*(normals.reshape(-1,1) * (params.reshape(-1,1) - beta * stats.reshape(-1,1) - alphas.reshape(1,-1))/gamma2).sum(axis=0)

def dL_dgamma2( params, stats, normals, alpha, beta, gamma2s):

    tmp = (params.reshape(-1,1) - beta*stats.reshape(-1,1) - alpha)**2 / gamma2s.reshape(1,-1)
    return 1/gamma2s.reshape(1,-1) * (normals.reshape(-1,1) * (1 - tmp)).sum(axis=0)
    
def alpha(params, stats, normals):
    
    N = normals.size    

    Eo  = (normals * params).sum()
    Eox = (normals * stats * params).sum()
    Ex2 = (normals * stats**2).sum()
    Ex  = (normals * stats).sum()
    E1  = normals.sum()
    
    #ahat = (normals * (Ex2 * params - Eox * stats)).sum()
    #ahat /= (E1 * Ex2 - Ex**2)
    
    ahat = (Eo - Eox/Ex2 * Ex) / (E1 - Ex**2/Ex2)
    
    return ahat

def beta(params, stats, normals, ahat=None):

    ahat = alpha(params, stats, normals) if ahat is None else ahat

    Eox = (normals * stats * params).sum()
    Ex2 = (normals * stats**2).sum()
    Ex  = (normals * stats).sum()
    
    bhat = (Eox - ahat * Ex) / Ex2
    
    return bhat
    
def gamma2(params, stats, normals, ahat=None, bhat=None):

    ahat = alpha(params, stats, normals) if ahat is None else ahat
    bhat = beta(params, stats, normals, ahat) if bhat is None else bhat

    gamma2hat = (normals*(params - ahat - bhat * stats )**2).sum() / normals.sum()
    
    return gamma2hat


def gamma2_bc(params, stats, normals, ahat=None, bhat=None):

    """ bias-corrected ? """

    ahat = alpha(params, stats, normals) if ahat is None else ahat
    bhat = beta(params, stats, normals, ahat) if bhat is None else bhat

    gamma2hat = (normals*(params - ahat - bhat * stats )**2).sum() 
    
    return gamma2hat


def analytic_div(out, eta2, nus, ksi2s):
    """ analytic correction of onedimensional Gaussians for proposal priors"""
    # assumes true prior to have zero mean!
    # INPUTS:
    # - out: 3D-tensor: 
    #        1st axis gives ksi2s (proposal variances), 
    #        2nd axis gives number of experiments/runs/fits
    #        3nd axis is size 2: out[i,j,0] Gaussian mean, out[i,j,0] Gaussian variance
    # - eta2:  prior variance (scalar)
    # - nus:   vector of proposal prior means
    # - ksi2s: vector of proposal prior variances
    
    # OUTPUTS
    # - out_: 3D tensor of proposal-corrected posterior means and variances
    
    out_ = np.empty_like(out)
    for i in range(out_.shape[0]):
        
        # precision and precision*mean
        P = 1/out[i,:,1]
        Pm = P * out[i,:,0]

        # multiply with prior
        P = P + 1/eta2
        Pm = Pm + 0/eta2

        # divide by proposal
        P = P - 1/ksi2s[i]
        Pm = Pm - nus[i]/ksi2s[i]

        out_[i,:,:] = np.vstack((Pm/P, 1/P)).T

    return out_


def test_setting(out_snpe, n_params, N, sig2, eta2, eps2, x0, ksi2s, nus, 
                 proposal_form, track_rp=False, df=None, marg=None, n_bins=50, if_plot=True,model='lin'):

    n_fits = out_snpe.shape[1]

    # compute target solution
    m = Gauss(dim=n_params, noise_cov=sig2)
    p = dd.Gaussian(m=0. * np.ones(n_params), 
                    S=eta2 * np.eye(n_params))
    post   = dd.Gaussian(m = np.ones(n_params) * eta2/(eta2+sig2)*x0[0], 
                         S=eta2 - eta2**2 / (eta2 + sig2) * np.eye(n_params)) 


    if if_plot:
        plt.figure(figsize=(4*len(ksi2s),8))
        m_m, m_v, M_m, M_v, hh_m, hh_v = np.inf,np.inf,-np.inf,-np.inf,-np.inf,-np.inf
    a,b = None,None
    
    for i in np.arange(len(ksi2s)):


        # compute proposal-posterior
        nu, ksi2 = nus[i], ksi2s[i]
        postpr = dd.Gaussian(m = np.ones(n_params) * (ksi2/(ksi2+sig2)*x0[0] + sig2/(ksi2+sig2)*nu), 
                             S=ksi2 - ksi2**2 / (ksi2 + sig2) * np.eye(n_params))


        # set up importance weight computation
        eta2p = 1/(1/eta2 - 1/ksi2)
        Sig_y = np.array([[eps2,0], [0,eta2p]])    
        mu_y = np.array([ [x0[0]], [eta2/(eta2-ksi2)*nu]])

        s = ds.Identity()
        pbar = progressbar(total=n_fits)
        desc = 'repeated fits'
        pbar.set_description(desc)
        with pbar:
            for idx_seed in range(n_fits):

                #print( str(idx_seed) + '/' + str(n_fits) )
                
                # excessive fixating of random seeds
                seed = 42 + idx_seed
                if proposal_form == 'normal':
                    ppr = dd.Gaussian(m=nu * np.ones(n_params), 
                                     S=ksi2 * np.eye(n_params),
                                     seed=seed)
                elif proposal_form == 'studentT':
                    ppr = dd.StudentsT(m=nu * np.ones(n_params), 
                                       S=ksi2 * np.eye(n_params), # * (df-2.)/df,
                                       dof=df,
                                       seed=seed)    
                elif proposal_form == 'unif':
                    a = nu - marg/2. * np.sqrt(ksi2)
                    b = nu + marg/2. * np.sqrt(ksi2)
                    ppr = dd.Uniform(lower=a, upper=b, seed=seed)
                else:
                    raise NotImplementedError

                if model=='lin':
                    m = Gauss(dim=n_params, noise_cov=sig2, seed=seed)
                elif model=='log':
                    m = LogGauss(dim=n_params, noise_cov=sig2, seed=seed)

                g = dg.Default(model=m, prior=ppr, summary=s)


                # gen data
                data = g.gen(N, verbose=False)
                params, stats = data[0].reshape(-1), data[1].reshape(-1)

                # 'fit' MDN
                normals = get_weights(proposal_form, eta2, ksi2, eps2, x0, nu, stats, params, df=df, a=a,b=b) if track_rp else np.ones(N)/N
                ahat =       alpha(params, stats, normals)
                bhat =        beta(params, stats, normals, ahat)
                gamma2hat = gamma2(params, stats, normals, ahat, bhat)

                # 'evaluate' MDN
                mu_hat   = ahat + bhat * x0
                sig2_hat = gamma2hat

                out_snpe[i,idx_seed,:] = (mu_hat, sig2_hat, normals.mean(), normals.min(), normals.max())
                pbar.update(1)

        if if_plot:

            post_disp = post if track_rp else postpr

            plt.subplot(len(ksi2s), 2, 2*i+1)
            m_m, M_m = np.min((m_m, out_snpe[i,:,0].min())), np.max((M_m, out_snpe[i,:,0].max()))
            plt.hist(out_snpe[i,:,0], bins=np.linspace(m_m, M_m, n_bins), normed=True)
            hh_m = np.max((hh_m, plt.axis()[3]))
            plt.plot([post_disp.mean, post_disp.mean], [0, hh_m], 'r', linewidth=2)
            #plt.plot(out_snpe[i,:,0].mean() + out_snpe[i,:,0].std()*np.array([-1,-1]), [0, hh_m/2], 'g', linewidth=2)
            plt.plot(out_snpe[i,:,0].mean() + out_snpe[i,:,0].std()*np.array([0,0]), [0, hh_m/2], 'g', linewidth=2)
            #plt.plot(out_snpe[i,:,0].mean() + out_snpe[i,:,0].std()*np.array([1,1]), [0, hh_m/2], 'g', linewidth=2)
            #plt.plot(out_snpe[i,:,0].mean() + out_snpe[i,:,0].std()*np.array([-1,1]), [ hh_m/2, hh_m/2], 'g', linewidth=2)
            plt.ylabel('xi^2/eta^2 = ' + str(ksi2/eta2) )

            plt.subplot(len(ksi2s),2, 2*i+2)
            m_v, M_v = np.min((m_v, out_snpe[i,:,1].min())), np.max((M_v, out_snpe[i,:,1].max()))
            plt.hist(out_snpe[i,:,1], bins=np.linspace(m_v, M_v, n_bins), normed=True)
            hh_v = np.max((hh_v, plt.axis()[3]))
            plt.plot([post_disp.std**2, post_disp.std**2], [0, hh_v], 'r', linewidth=2)
            #plt.plot(out_snpe[i,:,1].mean() + out_snpe[i,:,1].std()*np.array([-1,-1]), [0, hh_v/2], 'g', linewidth=2)
            plt.plot(out_snpe[i,:,1].mean() + out_snpe[i,:,1].std()*np.array([0,0]), [0, hh_v/2], 'g', linewidth=2)
            #plt.plot(out_snpe[i,:,1].mean() + out_snpe[i,:,1].std()*np.array([1,1]), [0, hh_v/2], 'g', linewidth=2)
            #plt.plot(out_snpe[i,:,1].mean() + out_snpe[i,:,1].std()*np.array([-1,1]), [ hh_v/2, hh_v/2], 'g', linewidth=2)
            #plt.ylabel('posterior variance')


    if if_plot:

        plt.subplot(len(ksi2s),2,1)
        plt.title('posterior mean')
        plt.subplot(len(ksi2s),2,2)
        plt.title('posterior variance')

        for i in range(len(ksi2s)):
            plt.subplot(len(ksi2s),2, 2*i+1)
            plt.axis([m_m, M_m, 0, hh_m])
            plt.subplot(len(ksi2s),2, 2*i+2)
            plt.axis([m_v, M_v, 0, hh_v])
        plt.show()

    return out_snpe


class LogGauss(BaseSimulator):
    def __init__(self, dim=1, noise_cov=0.1, seed=None):
        """Gauss simulator

        Toy model that draws data from a distribution centered on theta with
        fixed noise.

        Parameters
        ----------
        dim : int
            Number of dimensions of parameters
        noise_cov : float
            Covariance of noise on observations
        seed : int or None
            If set, randomness is seeded
        """
        super().__init__(dim_param=dim, seed=seed)
        self.noise_cov = noise_cov*np.eye(dim)

    @copy_ancestor_docstring
    def gen_single(self, param):
        # See BaseSimulator for docstring
        param = np.asarray(param).reshape(-1)
        assert param.ndim == 1
        assert param.shape[0] == self.dim_param

        sample = np.exp(dd.Gaussian(m=param, S=self.noise_cov,
                             seed=self.gen_newseed()).gen(1))

        return {'data': sample.reshape(-1)}
