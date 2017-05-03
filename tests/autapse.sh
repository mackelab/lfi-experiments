#/bin/sh

COMMON='--samples 500 --rep 5 --seed 1 --svi --iw-loss --sim-kwargs seed_obs=1,seed_input=42 --train-kwargs observe="weights",reg_scale=0.1 --val 100'


cd ..
python run.py autapse autapse_test_iw --rep 2 --iw-loss ${COMMON}
