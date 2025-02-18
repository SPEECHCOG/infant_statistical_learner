#!/bin/sh
export PATH="/projappl/project_2001315/khazar/con/vf/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

data_root=$1
exp_dir="../../exp6M/expS3/"
pretrained_root="../../../../hubertAndDINO"
twd="../../twd/"

python \
../run_spokencoco.py \
--image_type "normal" \
--subset "subset3" \
--data_root ${data_root} \
--trained_weights_dir ${twd} \
--exp_dir ${exp_dir} \
--load_pretrained_vit ${pretrained_root} \
--batch_size 64 \
--val_batch_size 64 \
--val_cross_batch_size 100 \
--n_epochs 100 \
--n_print_steps 59 \
--n_val_steps 590 \
--lr 0.0001 \
--warmup_fraction 0.1 \
--vit_arch 'vitsmall' \
--vit_patch_size 8 \
--vit_checkpoint_key 'teacher' \
--normalize \
--xtrm_layers 1 \
--trm_layers 6 \
--fine_matching_weight 0.0 \
--coarse_matching_weight 1.0 \
--libri_w2v2_weight 0.0 \
--caption_w2v2_weight 1.0 \
--feature_grad_mult 1.0 \
--trim_mask \
--layer_use 7 \

