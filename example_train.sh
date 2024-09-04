# example script for training CuNeRF

scale=1
file="/home/simtech/Qiming/kits19/data/case_00032/case_00032.nii.gz"
python run.py CuNeRFx$scale --cfg configs/example.yaml --scale $scale --mode train --file $file --save_map --resume --multi_gpu