#/bin/sh

# COMMON='--samples 2500 --rep 10 --svi --sim-kwargs seed_obs=1,seed_input=42'
COMMON1='--samples 500 --rep 15 --iw-loss --prior-alpha 0.05 --svi --sim-kwargs len_filter=1,seed_obs=1,seed_input=42'
# COMMON2='--samples 500 --rep 15 --rnn 2 --iw-loss --prior-alpha 0.05 --svi --sim-kwargs len_filter=1,seed_obs=1,seed_input=42 --train-kwargs observe='lasagne_out''

cd ..

# python run.py glm 10sta_2500_seed1 --seed 1 ${COMMON}
# python run.py glm 10sta_2500_seed2 --seed 2 ${COMMON}
# python run.py glm 10sta_2500_seed3 --seed 3 ${COMMON}
# python run.py glm 10sta_2500_seed4 --seed 4 ${COMMON}
# python run.py glm 10sta_2500_seed5 --seed 5 ${COMMON}

python run.py glm 2sta_500_iwloss_seed1 --seed 1 ${COMMON1}
python run.py glm 2sta_500_iwloss_seed2 --seed 2 ${COMMON1}
python run.py glm 2sta_500_iwloss_seed3 --seed 3 ${COMMON1}

# python run.py glm 2sta_500_iwloss_rnn_seed1 --seed 1 ${COMMON2}
# python run.py glm 2sta_500_iwloss_rnn_seed2 --seed 2 ${COMMON2}
# python run.py glm 2sta_500_iwloss_rnn_seed3 --seed 3 ${COMMON2}
