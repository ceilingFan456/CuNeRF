# example script for training CuNeRF
export CUDA_VISIBLE_DEVICES=0
scale=2
file="/home/simtech/Qiming/kits19/data/case_00010/case_00010.nii.gz"
python run.py CuNeRFx$scale --cfg configs/vcube-scale.yaml --scale $scale --mode train --file $file --save_map --resume --multi_gpu

## case 00000 for single
## imaging for coarse + fine