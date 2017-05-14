#/bin/sh
cd ..
cd ..

# 1-d MoG
COMMON='--enqueue cpu --nb --svi --units 20'
OURS='--iw-loss --prior-alpha 0.1 --train-kwargs reg_scale=0.1,n_minibatch=50'
EFREE='--train-kwargs n_minibatch=50'
for i in `seq 1 100`;
  do
    python run.py mog snpe_11_s$i --samples 1000,2000 --rep 1,1 --seed $i ${COMMON} ${OURS}
    python run.py mog snpe_21_s$i --samples 1000,1000,2000 --rep 2,1 --seed $i ${COMMON} ${OURS}
    python run.py mog snpe_31_s$i --samples 1000,1000,1000,2000 --rep 3,1 --seed $i ${COMMON} ${OURS}
    python run.py mog snpe_41_s$i --samples 1000,1000,1000,1000,2000 --rep 4,1 --seed $i ${COMMON} ${OURS}
    python run.py mog snpe_51_s$i --samples 1000,1000,1000,1000,1000,2000 --rep 5,1 --seed $i ${COMMON} ${OURS}

    python run.py mog efree_11_s$i --samples 1000,2000 --rep 1,1 --seed $i ${COMMON} ${EFREE}
    python run.py mog efree_21_s$i --samples 1000,1000,2000 --rep 2,1 --seed $i ${COMMON} ${EFREE}
    python run.py mog efree_31_s$i --samples 1000,1000,1000,2000 --rep 3,1 --seed $i ${COMMON} ${EFREE}
    python run.py mog efree_41_s$i --samples 1000,1000,1000,1000,2000 --rep 4,1 --seed $i ${COMMON} ${EFREE}
    python run.py mog efree_51_s$i --samples 1000,1000,1000,1000,1000,2000 --rep 5,1 --seed $i ${COMMON} ${EFREE}
  done

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
