# example script for training CuNeRF

scale=2
file="/home/simtech/Qiming/kits19/data/case_00069/imaging.nii.gz"
python run.py CuNeRFx$scale --cfg configs/example.yaml --scale $scale --mode train --file $file --save_map --resume
