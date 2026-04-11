

# Real-CD: Change Detection Under Real-World Complex Interference via Dynamic Distribution Correction
Leyuan Fang, Senior Member, IEEE, Yi Fang, Pedram Ghamisi, Senior Member, IEEE, and Qiang Liu

<div align="center">
    <img src="figures/framework.png" alt="framework" width="800"/>
</div>


# Getting Started

## step1: Environment Setup:

To get started, we recommend setting up a conda environment and installing dependencies via pip. Use the following commands to set up your environment

```bash
conda create -n realcd python==3.10
conda activate realcd
pip install -r requirements.txt
```
## step2: Our Change Detection Benchmark :

Our inference simulated benchmark based on DSIFN, SYSU, CLCD can be access at 

## step3: Model Training:

To train ChangeFormer on those datasets, 
you should run code file```scripts/run_train_ChangeFormer_xxx.sh```
, change your own data path and params.

Here, we provide our checkpoints for effective code recall.


## step4: Model Test Time Adaptation
Our work is tested on our change detection dataset DSIFN-C, SYSU-C, CLCD-C, the core file is ```tta_methods/Ours```

you should run code file and change your own data path and params.

```bash
bash scripts/eval_ChangeFormer_xxx_all.sh
```

the metric results are saved in ```ckpt``` folder and final prediction are saved at ```vistualization``` folder

## Moreover 
if you want to change the data path or model settings, please go to ``` scripts``` folder.


## Citation


## Acknowledgment

This code is mainly built upon [ChangeFormer](https://github.com/wgcban/ChangeFormer) and [Online Test Time adaptation](https://github.com/mariodoebler/test-time-adaptation) repositories.

