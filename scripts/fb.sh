#!/bin/sh
source activate fastvgs
export CUDA_VISIBLE_DEVICES=0,1,2,3

data_root=$1
fb_w2v2_weights_fn="../../model/wav2vec_small.pt"
exp_dir="../../expFB/"
libri_fn_root="../../datavf/ssl_sub2_root/"
pretrained_root="../../hubertAndDINO"
twd="../../twd/"

python \
../run_spokencoco.py \
--image_type "normal" \
--subset "all" \
--data_root ${data_root} \
--exp_dir ${exp_dir} \
--fb_w2v2_weights_fn ${fb_w2v2_weights_fn} \
--libri_fn_root ${libri_fn_root} \
--load_pretrained_vit ${pretrained_root} \
--batch_size 4 \
--val_batch_size 32 \
--val_cross_batch_size 100 \
--n_epochs 100 \
--n_print_steps 40 \
--n_val_steps 2000 \
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

