#!/usr/bin/env bash
gpus=0
net_G=ChangeFormerV6 #This is the best version
split=test
checkpoints_root=./ckpt/
project_name=CD_ChangeFormerV6_SYSU-256_b16_lr0.0001_adamw_train_val_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256
checkpoint_name=best_ckpt.pt
img_size=256
embed_dim=256 #Make sure to change the embedding dim (best and default = 256)

data_root=/home/hnu/E/0ASHARE/dataset/CD
data_name=SYSU-256

for conf in 'source' 'ttta'
do
  for i in 'Gauss' 'Fog' 'Snow' 'Motionblur' 'Impulse'
  do
    for k in 'Gauss' 'Fog' 'Snow' 'Motionblur' 'Impulse'
    do
      data_name=SYSU-${i}-${k}-256
      python eval_cd.py --cfg cfgs/SYSU/${conf}.yaml --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --data_root ${data_root}
    done
  done
  mv ${checkpoints_root}/${project_name}/log_test.txt \
    ${checkpoints_root}/${project_name}/SYSU_log_test_${conf}.txt
done

