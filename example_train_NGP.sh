# example script for training CuNeRF

scale=2
file="/home/simtech/Qiming/kits19/data/case_00004/case_00004.nii.gz"
python run.py CuNeRFx$scale --cfg configs/NGP.yaml --scale $scale --mode train --file $file --save_map --resume --multi_gpu

## case 00000 for single
## imaging for coarse + fine