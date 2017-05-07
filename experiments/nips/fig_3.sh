#/bin/sh
cd ..
cd ..

## HH murray
COMMON1='--samples 500 --rep 5 --svi --sim-kwargs seed_obs=1,seed_input=42,cached_pilot=False,cached_sims=False --val 500'
python run.py hh hh_500_svi_seed1 --seed 1 ${COMMON1}
python run.py hh hh_500_svi_seed2 --seed 2 ${COMMON1}
python run.py hh hh_500_svi_seed3 --seed 3 ${COMMON1}

## HH ours
COMMON2='--samples 500 --rep 5 --svi --accumulate-data --iw-loss --sim-kwargs seed_obs=1,seed_input=42,cached_pilot=False,cached_sims=False,cython=True --train-kwargs observe="weights",reg_scale=0.1 --val 500'
python run.py hh hh_500_iwloss_svi_inc_data_seed1 --seed 1 ${COMMON2}
python run.py hh hh_500_iwloss_svi_inc_data_seed2 --seed 2 ${COMMON2}
python run.py hh hh_500_iwloss_svi_inc_data_seed3 --seed 3 ${COMMON2}

## genetic algorithm (IBEA)


## real data
