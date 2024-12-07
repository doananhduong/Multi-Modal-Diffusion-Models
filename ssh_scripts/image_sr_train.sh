#################64 x 64 -> 256 x 256##########
MODEL_FLAGS="--sr_attention_resolutions 8,16,32  --large_size 256  
--small_size 64 --sr_learn_sigma True --sr_class_cond False
--sr_num_channels 192 --sr_num_heads 4 --sr_num_res_blocks 2 
--sr_resblock_updown True --use_fp16 True --sr_use_scale_shift_norm True"

DIFFUSION_FLAGS="--sr_diffusion_steps 1000 --noise_schedule linear" # --use_kl True

TRAIN_FLAGS="--lr 1e-4 --batch_size 4 --devices 0 --log_interval 100 --sample_fn ddpm 
 --save_interval 10000 --num_workers 8 --frame_gap 1 
 --use_db False --resume_checkpoint /root/MM-Diffusion/outputs/MM-Diffusion/sr-image-train/model090000.pt" #--schedule_sampler loss-second-moment --resume_checkpoint models/256x256_diffusion_uncond.pt

NUM_GPUS=1
DATA_DIR="/root/environmental_png/train"
OUT_DIR="/root/MM-Diffusion/outputs/MM-Diffusion/sr-image-train/"

mpiexec -n $NUM_GPUS python py_scripts/image_sr_train.py --data_dir $DATA_DIR --output_dir ${OUT_DIR} $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
