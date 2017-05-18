#/bin/sh
cd ..
cd ..

COMMON='--rep 2 --samples 1000 --seed 1'

python run_numerical.py sqrt notebooks/posterior --overwrite --no-random-samples

# python run.py sqrt testd1iwsvi --iw-loss --loss-calib 0.3 --svi ${COMMON}
# python run.py sqrt testd1iwsvi --iw-loss --loss-calib 10 --svi ${COMMON}
python run.py sqrt fig2bv1 --samples 5000 --rep 3
