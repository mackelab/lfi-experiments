#/bin/sh
cd ..
cd ..

# autapse model
python run.py autapse no_lr_s1 --samples 2000 --sim-kwargs pilot_samples=100 --seed 3 --iw-loss --debug --rep 10
python run.py autapse lr_s1 --bad-data --samples 2000 --sim-kwargs pilot_samples=100 --seed 3 --iw-loss --debug --rep 10
