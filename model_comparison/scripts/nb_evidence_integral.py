import numpy as np
import scipy.stats
import scipy.integrate
import time

def nb_evidence_integral(x, ks, thetas, integrant, log=False):

    k_grid, th_grid = np.meshgrid(ks, thetas)

    grid_values = np.zeros((thetas.size, ks.size))

    for i in range(thetas.shape[0]):
        for j in range(ks.shape[0]):
            grid_values[i, j] = integrant(k_grid[i, j], th_grid[i, j], x)

    integral = np.trapz(np.trapz(grid_values, x=thetas, axis=0), x=ks, axis=0)

    return np.log(integral) if log else integral


def nbinom_indirect_pmf(x, k, theta):
    gamma_pdf = scipy.stats.gamma(a=k, scale=theta)
    a = float(gamma_pdf.ppf(1e-8))
    b = float(gamma_pdf.ppf(1 - 1e-8))

    fun = lambda lam, k, theta, x: scipy.stats.poisson.pmf(x, mu=lam) * scipy.stats.gamma.pdf(lam, a=k, scale=theta)

    pmf_values = []
    for ix in x.squeeze():
        # integrate over all lambdas
        pmf_value, rr = scipy.integrate.quad(func=fun, a=a, b=b, args=(k, theta, ix), epsrel=1e-10)
        pmf_values.append(pmf_value)

    return pmf_values


def nbinom_pmf(k, r, p):
    k = k.squeeze()
    res = scipy.special.binom(k + r - 1, k) * np.power(p, k) * np.power(1-p, r)
    return res


def nb_evidence_integrant_indirect(k, theta, x):
    pk = prior_k.pdf(k)
    ptheta = prior_theta.pdf(theta)

    value = np.log(nbinom_indirect_pmf(x, k, theta)).sum() + np.log(pk) + np.log(ptheta)

    return np.exp(value)


def nb_evidence_integrant_direct(r, p, x):

    # get prior pdf values
    pk = prior_k.pdf(r)
    pp = np.power(1 - p, -2) * prior_theta.pdf(p / (1 - p))

    # take product (log sum) over samples and weight with prior probs
    value = np.log(nbinom_pmf(x, r, p)).sum() + np.log(pk) + np.log(pp)

    return np.exp(value)


def sample_poisson_gamma_mixture(prior1, prior2, n_samples, sample_size, seed=None):

    # set the seed
    np.random.seed(seed)

    thetas = []
    samples = []
    lambs = []

    for sample_idx in range(n_samples):

        # for every sample, get a new gamma prior
        thetas.append([prior1.rvs(), prior2.rvs()])
        gamma_prior = scipy.stats.gamma(a=thetas[sample_idx][0], scale=thetas[sample_idx][1])

        # now for every data point in the sample, to get NB, sample from that gamma prior into the poisson
        sample = []
        ls = []
        for ii in range(sample_size):
            ls.append(gamma_prior.rvs())
            sample.append(scipy.stats.poisson.rvs(ls[ii]))

        # add data set to samples
        samples.append(sample)
        lambs.append(ls)

    return np.array(thetas), np.array(samples), np.array(lambs)


n_steps = 1000
sample_size = 3
n_samples = 1
seed = 1
time_stamp = time.strftime('%Y%m%d%H%M_')
figure_folder = '../figures/'

# set prior parameters
# set the shape or scale of the Gamma prior for the Poisson model
k1 = 9.0
# set the shape and scale of the prior on the shape of the Gamma for the mixture to be broad
theta2 = 2.0
k2 = 5.
# set the shape and scale of the prior on the scale of the Gamma for the mixture to be small
# this will make the variance and could be the tuning point of the amount of overdispersion / difficulty
theta3 = 1.0
k3 = 1

# then the scale of the Gamma prior for the Poisson is given by
theta1 = (k2 * theta2 * k3 * theta3) / k1

# get analytical means
mean_ana_poi = k1 * theta1
mean_ana_nb = k2 * k3 * theta2 * theta3

# set the priors
prior_k = scipy.stats.gamma(a=k2, scale=theta2)
prior_theta = scipy.stats.gamma(a=k3, scale=theta3)

# draw sample(s)
params_nb, X, lambs = sample_poisson_gamma_mixture(prior_k, prior_theta, n_samples, sample_size, seed=seed)

# set up a grid of values around the priors
# take grid over the whole range of the priors
k0 = scipy.stats.gamma.ppf(1e-8, a=k2)
k1 = scipy.stats.gamma.ppf(1 - 1e-8, a=k2)

theta0 = scipy.stats.gamma.ppf(1e-8, a=k3)
theta1 = scipy.stats.gamma.ppf(1 - 1e-8, a=k3)

evianas = []

for x in X:
    (eviana, err) = scipy.integrate.dblquad(func=nb_evidence_integrant_direct,
                                            a=theta0 / (1 + theta0),
                                            b=theta1 / (1 + theta1),
                                            gfun=lambda x: k0, hfun=lambda x: k1, args=[x])
    evianas.append(eviana)

print(evianas)

evianas = []
for x in X:
    (eviana, err) = scipy.integrate.dblquad(func=nb_evidence_integrant_indirect, a=theta0, b=theta1,
                                            gfun=lambda x: k0, hfun=lambda x: k1, args=[x])
    evianas.append(eviana)

print(evianas)