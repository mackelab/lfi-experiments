python run.py hh lp1s1 --rep 0,5 --samples 25000 --svi --iw-loss  --sim-kwargs prior_extent=True,pilot_samples=5000,seed_obs=1 --units [25,25,25] --seed 1 --enqueue cpu --train-kwargs reg_init=True
python run.py hh lp1s2 --rep 0,5 --samples 25000 --svi --iw-loss  --sim-kwargs prior_extent=True,pilot_samples=5000,seed_obs=1 --units [25,25,25] --seed 2 --enqueue cpu --train-kwargs reg_init=True

python run.py hh lp2s1 --rep 0,0,5 --samples 25000 --svi --iw-loss --heavy-tails --sim-kwargs prior_extent=True,pilot_samples=5000,seed_obs=1 --units [25,25,25] --seed 1 --enqueue cpu --train-kwargs reg_init=True
python run.py hh lp2s2 --rep 0,0,5 --samples 25000 --svi --iw-loss --heavy-tails --sim-kwargs prior_extent=True,pilot_samples=5000,seed_obs=1 --units [25,25,25] --seed 2 --enqueue cpu --train-kwargs reg_init=True

python run.py hh lp3s1 --rep 0,0,0,5 --samples 25000 --svi --iw-loss --heavy-tails --sim-kwargs prior_extent=True,pilot_samples=5000,seed_obs=1 --units [40,40,40] --seed 1 --enqueue cpu --train-kwargs reg_init=True
python run.py hh lp3s2 --rep 0,0,0,5 --samples 25000 --svi --iw-loss --heavy-tails --sim-kwargs prior_extent=True,pilot_samples=5000,seed_obs=1 --units [40,40,40] --seed 2 --enqueue cpu --train-kwargs reg_init=True
