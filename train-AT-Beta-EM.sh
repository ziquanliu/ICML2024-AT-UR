#!/bin/sh
# export CUDA_VISIBLE_DEVICES=0
lr=0.03
wd=0.0001
lambda_KD_robust=0.5
lambda_dis=6.0

beta_dist_alpha=1.1
beta_dist_beta=5.0
lambda_ent=0.3

python AT-Beta-EM.py --data_root '/home/data' --model_root './adv_pretrain_R101_lr_'$lr'_wd_'$wd'_alpha_'$beta_dist_alpha'_beta_'$beta_dist_beta'_lambda_ent_'$lambda_ent'_epoch_60_v2_trial_1' -w $wd -e 0.0314 --lambda_distil $lambda_dis --learning_rate $lr -p 'linf' --adv_train --affix 'linf' --log_root './adv_pretrain_R101_lr_'$lr'_wd_'$wd'_alpha_'$beta_dist_alpha'_beta_'$beta_dist_beta'_lambda_ent_'$lambda_ent'_epoch_60_v2_trial_1_log' --gpu '0' -m_e 60 --model-path './r50_imagenet_linf_4.pt' --num_classes 100 --beta_dist_alpha $beta_dist_alpha --beta_dist_beta $beta_dist_beta --lambda_ent $lambda_ent
