#/bin/sh

COMMON1='--samples 5000 --rep 5 --svi --sim-kwargs len_filter=9,seed_obs=1,seed_input=42 --val 500'
COMMON2='--samples 5000 --rep 5 --svi --units [50,50] --loss-calib 10,2 --early-stopping --iw-loss --sim-kwargs len_filter=9,seed_obs=1,seed_input=42 --train-kwargs observe="weights",reg_scale=0.1 --val 500'


cd ..

python run.py glm 10sta_5000_svi_seed1 --seed 1 ${COMMON1}
python run.py glm 10sta_5000_svi_seed2 --seed 2 ${COMMON1}
python run.py glm 10sta_5000_svi_seed3 --seed 3 ${COMMON1}

python run.py glm 10sta_5000_iwloss_svi_inc_data_seed1 --seed 1 ${COMMON2}
python run.py glm 10sta_5000_iwloss_svi_inc_data_seed2 --seed 2 ${COMMON2}
python run.py glm 10sta_5000_iwloss_svi_inc_data_seed3 --seed 3 ${COMMON2}
