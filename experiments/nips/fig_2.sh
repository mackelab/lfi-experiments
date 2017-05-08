#/bin/sh
cd ..
cd ..

# MoG

# ...

# GLM

## 10d murray
COMMON1='--samples 5000 --rep 5 --svi --sim-kwargs len_filter=9,seed_obs=1,seed_input=42 --val 500'
python run.py glm 10sta_5000_svi_seed1 --seed 1 ${COMMON1}
python run.py glm 10sta_5000_svi_seed2 --seed 2 ${COMMON1}
python run.py glm 10sta_5000_svi_seed3 --seed 3 ${COMMON1}

## 10d ours
COMMON2='--samples 5000 --rep 5 --svi --accumulate-data --units [50,50] --loss-calib 10,2 --early-stopping --iw-loss --sim-kwargs len_filter=9,seed_obs=1,seed_input=42 --train-kwargs reg_scale=0.0 --val 500'
python run.py glm 10sta_5000_iwloss_svi_inc_data_seed1 --seed 1 ${COMMON2}
python run.py glm 10sta_5000_iwloss_svi_inc_data_seed2 --seed 2 ${COMMON2}
python run.py glm 10sta_5000_iwloss_svi_inc_data_seed3 --seed 3 ${COMMON2}

## ess-mcmc
python run_mcmc.py glm 10sta_5000_svi_seed1
python run_mcmc.py glm 10sta_5000_svi_seed2
python run_mcmc.py glm 10sta_5000_svi_seed3
python run_mcmc.py glm 10sta_5000_iwloss_svi_inc_data_seed1
python run_mcmc.py glm 10sta_5000_iwloss_svi_inc_data_seed2
python run_mcmc.py glm 10sta_5000_iwloss_svi_inc_data_seed3
