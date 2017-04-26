#/bin/sh

# COMMON='--samples 2500 --svi --sim-kwargs seed_obs=1,seed_input=42,cached_pilot=0,cached_sims=0'
COMMON1='--samples 500 --iw-loss --svi --sim-kwargs seed_obs=1,seed_input=42,cached_pilot=0,cached_sims=0'

cd ..

# python run.py glm 10sta_2500_seed1 --rep 10 --seed 1 ${COMMON}
# python run.py glm 10sta_2500_seed2 --rep 10 --seed 2 ${COMMON}
# python run.py glm 10sta_2500_seed3 --rep 10 --seed 3 ${COMMON}
# python run.py glm 10sta_2500_seed4 --rep 10 --seed 4 ${COMMON}
# python run.py glm 10sta_2500_seed5 --rep 10 --seed 5 ${COMMON}

# python run.py glm 2sta_500_iwloss_seed1 --rep 15 --seed 1 ${COMMON1}
# python run.py glm 2sta_500_iwloss_seed2 --rep 15 --seed 2 ${COMMON1}
# python run.py glm 2sta_500_iwloss_seed3 --rep 15 --seed 3 ${COMMON1}

python run.py glm 2sta_500_iwloss_rnn_seed1 --rnn 2 --rep 15 --seed 1 ${COMMON1}
python run.py glm 2sta_500_iwloss_rnn_seed2 --rnn 2 --rep 15 --seed 2 ${COMMON1}
python run.py glm 2sta_500_iwloss_rnn_seed3 --rnn 2 --rep 15 --seed 3 ${COMMON1}
