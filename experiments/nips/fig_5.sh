#/bin/sh
cd ..
cd ..

python run.py hh shrink_nips_v1_60ms --rep 1 --rnn 25 --samples 1000 --seed 2 --svi --sim-kwargs cython=True,duration=60,seed_input=3,seed_obs=42,step_current=False,cached_pilot=False,cached_sims=False --train-kwargs n_iter=500,n_minibatch=250 --device cuda0 --enqueue gpu
python run.py hh shrink_nips_v1_120ms --rep 1 --rnn 25 --samples 1000 --seed 2 --svi --sim-kwargs cython=True,duration=120,seed_input=3,seed_obs=42,step_current=False,cached_pilot=False,cached_sims=False --train-kwargs n_iter=500,n_minibatch=250 --device cuda0 --enqueue gpu
python run.py hh shrink_nips_v1_240ms --rep 1 --rnn 25 --samples 1000 --seed 2 --svi --sim-kwargs cython=True,duration=240,seed_input=3,seed_obs=42,step_current=False,cached_pilot=False,cached_sims=False --train-kwargs n_iter=500,n_minibatch=250 --device cuda0 --enqueue gpu
