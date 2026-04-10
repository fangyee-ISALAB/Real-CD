#!/usr/bin/env bash
gpus=0
net_G=ChangeFormerV6 #This is the best version
split=test
vis_root=/home/hnu/E/0ASHARE/CD/Changformer/ChangeFormer/media/lidan/ChangeFormer/vis
checkpoints_root=/home/hnu/E/0ASHARE/CD/Changformer/ckpt/
project_name=CD_ChangeFormerV6_DSIFN_b16_lr0.00006_adamw_train_test_200_linear_ce_multi_train_True_multi_infer_False_shuffle_AB_False_embed_dim_256
checkpoint_name=best_ckpt.pt
img_size=256
embed_dim=256 #Make sure to change the embedding dim (best and default = 256)

data_root=/home/hnu/E/0ASHARE/dataset/CD
data_name=DSIFN-256
#for nums in '1' '2' '3' '4' '5'
for nums in '1'
do
  for conf in 'source' 'ttta' 'cmf' 'memo' 'cotta' 'sar' 'tent' 'roid' 'recap'
#  for conf in 'cmf' 'memo' 'cotta' 'tent' 'roid'
#  for conf in 'source'
#  for conf in 'recap'
  do
    data_name=DSIFN-256
    python eval_cd.py --cfg cfgs/DSIFN/${conf}.yaml --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --data_root ${data_root}
#    for i in 'Gauss' 'Fog' 'Snow' 'Motionblur' 'Impulse'
    for i in 'Gauss'
    do
#      for k in 'Gauss' 'Fog' 'Snow' 'Motionblur' 'Impulse'
      for k in 'Fog'
      do
        data_name=DSIFN-${i}-${k}-256
        python eval_cd.py --cfg cfgs/DSIFN/${conf}.yaml --split ${split} --net_G ${net_G} --embed_dim ${embed_dim} --img_size ${img_size} --checkpoints_root ${checkpoints_root} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name} --data_root ${data_root}
      done
    done
    mv ../../ckpt/${project_name}/log_test.txt \
      ../../ckpt/${project_name}/DSIFN_log_test_${conf}_${nums}.txt
  done
done