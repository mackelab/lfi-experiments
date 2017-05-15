#/bin/sh
cd ..
cd ..

## HH murray
COMMON1='--samples 1000 --rep 0,5 --svi --sim-kwargs seed_obs=1,seed_input=42,cached_pilot=False,cached_sims=False,cython=True --val 500 --enqueue all'
python run.py hh hh_1000_svi_cyth_seed1ncomp2 --seed 1 ${COMMON1}
python run.py hh hh_1000_svi_cyth_seed2ncomp2 --seed 2 ${COMMON1}
python run.py hh hh_1000_svi_cyth_seed3ncomp2 --seed 3 ${COMMON1}

## HH ours and genetic algorithm (IBEA)
COMMON2='--samples 1000 --train-kwargs reg_scale=0.1 --prior-alpha 0.1 --rep 0,5 --svi --iw-loss --genetic --sim-kwargs seed_obs=1,seed_input=42,cached_pilot=False,cached_sims=False,cython=True --enqueue all'
python run.py hh hh_1000_iwloss_svi_cyth_seed1_v4pa01ncomp2 --seed 1 ${COMMON2}
python run.py hh hh_1000_iwloss_svi_cyth_seed2_v4pa01ncomp2 --seed 2 ${COMMON2}
python run.py hh hh_1000_iwloss_svi_cyth_seed3_v4pa01ncomp2 --seed 3 ${COMMON2}


## real data
COMMON3='--samples 1000 --train-kwargs reg_scale=0.1 --prior-alpha 0.1 --rep 0,5 --svi --iw-loss --sim-kwargs seed_obs=1,seed_input=42,cached_pilot=False,cached_sims=False,obs_sim=False,cython=True --enqueue all'
python run.py hh hh_1000_iwloss_svi_cyth_realdata_seed1_v3pa01ncomp2 --seed 1 ${COMMON3}
python run.py hh hh_1000_iwloss_svi_cyth_realdata_seed2_v3pa01ncomp2 --seed 2 ${COMMON3}
python run.py hh hh_1000_iwloss_svi_cyth_realdata_seed3_v3pa01ncomp2 --seed 3 ${COMMON3}
