#!/bin/sh
export PATH="/projappl/project_2001315/khazar/con/vf/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0,1,2,3

data_root=$1
fb_w2v2_weights_fn="../../../../model/wav2vec_small.pt"
exp_dir="../../expnewl15/"
twd="../../twdp1/"
libri_fn_root="../../../../datavf/ssl6M_root/"
pretrained_root="../../../../hubertAndDINO"

python \
../run_spokencoco.py \
--ssl \
--data_root ${data_root} \
--exp_dir ${exp_dir} \
--libri_fn_root ${libri_fn_root} \
--resume \
--load_pretrained_vit ${pretrained_root} \
--libri_max_seq_len 15 \
--batch_size 22 \
--val_batch_size 22 \
--val_cross_batch_size 100 \
--n_epochs 100 \
--n_print_steps 100 \
--n_val_steps 3412 \
--lr 0.0001 \
--warmup_fraction 0.1 \
--vit_arch 'vitsmall' \
--vit_patch_size 8 \
--vit_checkpoint_key 'teacher' \
--normalize \
--xtrm_layers 1 \
--trm_layers 1 \
--fine_matching_weight 0.0 \
--coarse_matching_weight 1.0 \
--libri_w2v2_weight 0.0 \
--caption_w2v2_weight 1.0 \
--feature_grad_mult 0.1 \
--trim_mask \
--encoder_layers 12 \
--encoder_attention_heads 8 \
--layer_use 11 \

