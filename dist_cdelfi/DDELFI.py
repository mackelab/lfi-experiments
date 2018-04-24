import delfi.distribution as dd
import numpy as np
import theano.tensor as tt

from delfi.inference.BaseInference import BaseInference
from delfi.neuralnet.NeuralNet import NeuralNet
from delfi.neuralnet.Trainer import Trainer
from delfi.neuralnet.loss.regularizer import svi_kl_zero

from mogtrain import MoGTrainer, DefensiveDistribution, CroppedDistribution

class DDELFI(BaseInference):
    def __init__(self, generator, obs, prior_norm=False, pilot_samples=100,
                 n_components=1, reg_lambda=0.01, seed=None, verbose=True,
                 convert_to_T=None, prior_mixin=0, init_norm=False, 
                 reinit_weights=False, **kwargs):
        """Conditional density estimation likelihood-free inference (CDE-LFI)

        Implementation of algorithms 1 and 2 of Papamakarios and Murray, 2016.

        Parameters
        ----------
        generator : generator instance
            Generator instance
        obs : array
            Observation in the format the generator returns (1 x n_summary)
        prior_norm : bool
            If set to True, will z-transform params based on mean/std of prior
        pilot_samples : None or int
            If an integer is provided, a pilot run with the given number of
            samples is run. The mean and std of the summary statistics of the
            pilot samples will be subsequently used to z-transform summary
            statistics.
        n_components : int
            Number of components in final round (PM's algorithm 2)
        reg_lambda : float
            Precision parameter for weight regularizer if svi is True
        seed : int or None
            If provided, random number generator will be seeded
        verbose : bool
            Controls whether or not progressbars are shown
        kwargs : additional keyword arguments
            Additional arguments for the NeuralNet instance, including:
                n_hiddens : list of ints
                    Number of hidden units per layer of the neural network
                svi : bool
                    Whether to use SVI version of the network or not

        Attributes
        ----------
        observables : dict
            Dictionary containing theano variables that can be monitored while
            training the neural network.
        """
        super().__init__(generator, prior_norm=prior_norm,
                         pilot_samples=pilot_samples, seed=seed,
                         n_components=n_components,
                         verbose=verbose, **kwargs)

        self.n_components = n_components
        self.obs = obs
        self.prior_mixin = prior_mixin
        self.convert_to_T = convert_to_T

        if np.any(np.isnan(self.obs)):
            raise ValueError("Observed data contains NaNs")

        self.reg_lambda = reg_lambda
        self.reinit_weights = reinit_weights
        self.init_norm = init_norm

    def loss(self, N):
        """Loss function for training

        Parameters
        ----------
        N : int
            Number of training samples
        """
        loss = -tt.mean(self.network.lprobs)

        return loss

    def run(self, n_train=100, n_rounds=2, epochs=100, minibatch=50,
            monitor=None, impute_proposal=None, n_ensemble=None, **kwargs):
        """Run algorithm

        Parameters
        ----------
        n_train : int or list of ints
            Number of data points drawn per round. If a list is passed, the
            nth list element specifies the number of training examples in the
            nth round. If there are fewer list elements than rounds, the last
            list element is used.
        n_rounds : int
            Number of rounds
        epochs: int
            Number of epochs used for neural network training
        minibatch: int
            Size of the minibatches used for neural network training
        monitor : list of str
            Names of variables to record during training along with the value
            of the loss function. The observables attribute contains all
            possible variables that can be monitored
        kwargs : additional keyword arguments
            Additional arguments for the Trainer instance

        Returns
        -------
        logs : list of dicts
            Dictionaries contain information logged while training the networks
        trn_datasets : list of (params, stats)
            training datasets, z-transformed
        posteriors : list of posteriors
            posterior after each round
        """
        logs = []
        trn_datasets = []
        posteriors = []
        preds = []

        for r in range(1, n_rounds + 1):  # start at 1
            # if round > 1, set new proposal distribution before sampling
            if impute_proposal:
                self.generator.proposal = impute_proposal
            elif r > 1:
                # posterior becomes new proposal prior
                if len(posteriors) != 0:
                    posterior = posteriors[-1]
                else:
                    pred = self.compute_posterior(self.obs, **mog_kwargs)
                    posterior = pred['posterior']
                    
                proposal = posterior
                if self.convert_to_T:
                    proposal = proposal.convert_to_T(dofs=self.convert_to_T)
                    
                if self.prior_mixin:
                    proposal = DefensiveDistribution(proposal, self.generator.prior, alpha=self.prior_mixin, seed=self.gen_newseed())
                
                if isinstance(self.generator.prior, dd.Uniform) or isinstance(self.generator.prior, dd.LogUniform):
                    proposal = CroppedDistribution(proposal, self.generator.prior)
                    
                self.generator.proposal = proposal
                
            if r > 1 and self.reinit_weights:
                self.reinit_network()

            # number of training examples for this round
            if type(n_train) == list:
                try:
                    n_train_round = n_train[r-1]
                except:
                    n_train_round = n_train[-1]
            else:
                n_train_round = n_train

            if type(epochs) in (list,tuple):
                try:
                    epochs_round = epochs[r-1]
                except IndexError:
                    epochs_round = epochs[-1]
            else:
                epochs_round = epochs

            # draw training data (z-transformed params and stats)
            verbose = '(round {}) '.format(r) if self.verbose else False
            trn_data = self.gen(n_train_round, verbose=verbose)
            
            mog_kwargs = { k[4:] : kwargs[k] for k in kwargs if k.startswith("mog_") }
            trn_kwargs = { k : kwargs[k] for k in kwargs if not k.startswith("mog_") }
            
            if n_ensemble is None:
                trn_inputs = [self.network.params, self.network.stats]
                
                t = Trainer(self.network, self.loss(N=n_train_round),
                            trn_data=trn_data, trn_inputs=trn_inputs,
                            monitor=self.monitor_dict_from_names(monitor),
                            seed=self.gen_newseed(), **trn_kwargs)
                
                logs.append(t.train(epochs=epochs_round, minibatch=minibatch,
                                    verbose=verbose))

                trn_datasets.append(trn_data)
                pred = self.compute_posterior(self.obs, **mog_kwargs)

                preds.append(pred)
                posteriors.append(pred['posterior'])
            else:
                logs_r = []
                trn_data_r = []
                preds_r = []
                posteriors_r = []

                for i in range(n_ensemble):
                    self.reinit_network()
                    
                    trn_inputs = [self.network.params, self.network.stats]
                
                    t = Trainer(self.network, self.loss(N=n_train_round),
                                trn_data=trn_data, trn_inputs=trn_inputs,
                                monitor=self.monitor_dict_from_names(monitor),
                                seed=self.gen_newseed(), **trn_kwargs)

                    logs_r.append(t.train(epochs=epochs_round, minibatch=minibatch,
                                          verbose=verbose))

                    trn_data_r.append(trn_data)

                    pred = self.compute_posterior(self.obs, **mog_kwargs)

                    preds_r.append(pred)
                    posteriors_r.append(pred['posterior'])

                logs.append(logs_r)
                trn_datasets.append(trn_data_r)
                preds.append(preds_r)
                posteriors.append(posteriors_r)          

        return logs, trn_datasets, posteriors, preds

    def predict(self, **kwargs):
        raise NotImplementedError("Must call DDELFI.compute_posterior to obtain posteriors")
        
    def compute_posterior(self, x, nsamples=10000, nsteps=-1, **kwargs):
        """Predict posterior given x

        Parameters
        ----------
        x : array
            Stats for which to compute the posterior
        """
        if self.generator.proposal is None:
            # no correction necessary
            return { 'posterior' : super().predict(x) }  # via super
        else:
            qphi = super().predict(x)  # via super
            
            trainer = MoGTrainer(prop=self.generator.proposal, 
                                 prior=self.generator.prior, 
                                 qphi=qphi, 
                                 n_components=self.n_components, 
                                 nsamples=nsamples, 
                                 **kwargs)
            
            trainer.train(nsteps=nsteps)
            
            posterior = trainer.get_mog()
            return { 'posterior' : posterior, 'prior' : self.generator.prior, 'proposal' : self.generator.proposal, 'qphi' : qphi }
 
