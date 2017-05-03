#/bin/sh

COMMON='--samples 100 --rep 2 --seed 1 --svi --iw-loss --sim-kwargs seed_obs=1 --train-kwargs observe="weights",reg_scale=0.1 --val 100'


cd ..
python run.py autapse autapse_test_iw --iw-loss ${COMMON}
