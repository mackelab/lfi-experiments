# NIPS 2017

Experiments generating results and figures for NIPS submission.

To run, make sure to:
- Have `delfi` and `lfimodels` installed
- ...




#/bin/sh

# FIGURE 1
COMMON='--rep 2 --samples 1000 --seed 1'
python run_numerical.py sqrt notebooks/posterior --overwrite --no-random-samples
python run.py sqrt fig2bv1 --samples 5000 --rep 3


# FIGURE 2
python run.py mog fork_gt --rep 0,1 --samples 10000 --units 20 --nb --debug --svi --seed 1 --enqueue cpu --sim-kwargs bimodal=True
for i in `seq 1 1`;
  do
    #python run.py mog fork_v6_snpe_s$i --rep 1,2 --samples 250 --units 20 --nb --debug --svi --iw-loss --prior-alpha 0.0 --seed $i --enqueue cpu --train-kwargs reg_scale=0.1,reg_init=True --sim-kwargs bimodal=True
    #python run.py mog fork_v5_efree_s$i --rep 2,1 --samples 250 --units 20 --nb --debug --svi --seed $i --enqueue cpu --train-kwargs reg_scale=0.1 --sim-kwargs bimodal=True
  done


# FIGURE 3

## HH murray
COMMON1='--samples 1000 --rep 0,5 --svi --sim-kwargs seed_obs=1,seed_input=42,cached_pilot=False,cached_sims=False,cython=True --val 500 --enqueue all'
# python run.py hh hh_1000_svi_cyth_seed1ncomp2 --seed 1 ${COMMON1}
# python run.py hh hh_1000_svi_cyth_seed2ncomp2 --seed 2 ${COMMON1}
# python run.py hh hh_1000_svi_cyth_seed3ncomp2 --seed 3 ${COMMON1}

## HH ours and genetic algorithm (IBEA)
COMMON2='--samples 5000 --train-kwargs reg_scale=1,reg_init=True --prior-alpha 0 --rep 0,5 --svi --iw-loss --sim-kwargs seed_obs=1,seed_input=42,cached_pilot=False,cached_sims=False,cython=True --enqueue all'
python run.py hh hh_5000_iwloss_svi_cyth_seed1_v6 --seed 1 ${COMMON2}
python run.py hh hh_5000_iwloss_svi_cyth_seed2_v6 --seed 2 ${COMMON2}
#python run.py hh hh_5000_iwloss_svi_cyth_seed3_v6 --seed 3 ${COMMON2}

## HH ours and genetic algorithm (IBEA)
COMMON2='--samples 5000 --train-kwargs reg_scale=1,reg_init=True --prior-alpha 0 --rep 0,5 --svi --iw-loss --sim-kwargs seed_obs=1,seed_input=42,cached_pilot=False,cached_sims=False,cython=True,prior_extent=True --enqueue all'
python run.py hh hh_5000_iwloss_svi_cyth_seed1_v6 --seed 1 ${COMMON2}
python run.py hh hh_5000_iwloss_svi_cyth_seed2_v6 --seed 2 ${COMMON2}
python run.py hh hh_5000_iwloss_svi_cyth_seed3_v6 --seed 3 ${COMMON2}


## real data
COMMON3='--samples 1000 --train-kwargs reg_scale=0.1 --prior-alpha 0.1 --rep 0,5 --svi --iw-loss --sim-kwargs seed_obs=1,seed_input=42,cached_pilot=False,cached_sims=False,obs_sim=False,cython=True --enqueue all'
# python run.py hh hh_1000_iwloss_svi_cyth_realdata_seed1_v3pa01ncomp2 --seed 1 ${COMMON3}
# python run.py hh hh_1000_iwloss_svi_cyth_realdata_seed2_v3pa01ncomp2 --seed 2 ${COMMON3}
# python run.py hh hh_1000_iwloss_svi_cyth_realdata_seed3_v3pa01ncomp2 --seed 3 ${COMMON3}


# FIGURE 4

python run.py autapse no_lr_s1 --samples 2000 --sim-kwargs pilot_samples=100 --seed 3 --iw-loss --debug --rep 10
python run.py autapse lr_s1 --bad-data --samples 2000 --sim-kwargs pilot_samples=100 --seed 3 --iw-loss --debug --rep 10


# FIGURE 5

python run.py hh shrink_nips_v1_60ms --rep 1 --rnn 25 --samples 1000 --seed 2 --svi --sim-kwargs cython=True,duration=60,seed_input=3,seed_obs=42,step_current=False,cached_pilot=False,cached_sims=False --train-kwargs n_iter=500,n_minibatch=250 --device cuda0 --enqueue gpu
python run.py hh shrink_nips_v1_120ms --rep 1 --rnn 25 --samples 1000 --seed 2 --svi --sim-kwargs cython=True,duration=120,seed_input=3,seed_obs=42,step_current=False,cached_pilot=False,cached_sims=False --train-kwargs n_iter=500,n_minibatch=250 --device cuda0 --enqueue gpu
python run.py hh shrink_nips_v1_240ms --rep 1 --rnn 25 --samples 1000 --seed 2 --svi --sim-kwargs cython=True,duration=240,seed_input=3,seed_obs=42,step_current=False,cached_pilot=False,cached_sims=False --train-kwargs n_iter=500,n_minibatch=250 --device cuda0 --enqueue gpu
