
export LSF_DOCKER_SHM_SIZE=600g; bsub -q general -R 'gpuhost span[hosts=1] rusage[mem=600GB]' -M 800GB -gpu "num=8:gmodel=NVIDIAA40:j_exclusive=yes" -o 'logs/Brooklyn_4cond_with_streetviewgeomap_soft_8card_all0.5_fused_3layers_new.log' \
-a 'docker(continuumio/miniconda3)' 'cd /storage1/fs1/jacobsn/Active/user_x.zhexiao/Uni-ControlNet; conda init bash; conda run --no-capture-output -n unicontrol;  python src/train/train.py'
