#!/bin/bash

#SBATCH --job-name=train_sota      # Submit a job named "example"
#SBATCH --partition=a100        # 계산노드 종류 선택: a6000 or a100
#SBATCH --gres=gpu:a100.10gb:1        # Use 1 GPU or gpu:a100.5gb:1
#SBATCH --mem=10000
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=../logs/train_sota_05.out   # 스크립트 실행 결과 std output을 저장할 파일 이름
#SBATCH --mail-user=onlyou0416@snu.ac.kr
#SBATCH --mail-type=FAIL

ml purge
ml load cuda/11.3
eval "$(conda shell.bash hook)"  # Initialize Conda Environment
conda activate sbir_clip             # Activate your conda environment

srun python LN_prompt.py --exp_name=clip_split5_03 --n_prompts=1 --margin=0.3 --prompt_lr=1e-5 --clip_LN_lr=1e-5 \
                         --data_split=0.5 --workers=1 --batch_size=64 --check_val_every_n_epoch=1

# --ntasks-per-node=2

#--master_port=35500 --nproc_per_node=2
# srun python -m torch.distributed.launch main.py --num_workers 12 --train_ds_idx 0 4 6 7 --val_ds_idx 0 \
#         --src_mode 0 --trg_mode 1 --total_n_modes 2 --cross y --img_input_size 224 224 --use_vis_layer_norm y \
#         --epochs 100 --batch_size 64 --gradient_accumulation_steps 16 --lr 1e-4 --lr_sched cosine --dec_type 1 \
#         --neg_extended y --use_converted_trg n --cont_loss_ratio 10.0 --temperature 0.5 --task_prefix "caption: " "caption of image: " --img_enc_model vit --valid_partial_ratio 0.2
