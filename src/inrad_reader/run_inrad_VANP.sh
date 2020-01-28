#!/bin/bash
#SBATCH -A m1657
#SBATCH --job-name=run_inrad.py
#SBATCH -p regular
#SBATCH --ntasks=1
#SBATCH -C haswell
#SBATCH --mail-user=jingyi.chen@pnnl.gov
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --output=./log/run_inrad_VANP.out
#SBATCH --exclusive

# enter the folder
cd /global/homes/c/chen696/inrad_reader/inrad_reader/src/inrad_reader

# python environment
bash
source /global/common/cori/software/python/3.6-anaconda-5.2/bin/activate /global/homes/c/chen696/.conda/envs/myenv


#  start loop through selected files
date
radar_name="VANP"
date_str="20180706"

#for time_str in "001202" "071203" "092202" "105203" "122202" "174202" "190203" "202202" "232202"
for time_str in "001202" "071203"
do

	input_name="/project/projectdirs/m1657/zfeng/indian_radar/${radar_name}-RADAR/${date_str}/T_HAHA00_C_${radar_name}_${date_str}${time_str}"
	echo $input_name\*
	python inrad_to_cf.py -fg $input_name\* -c example_config.json -og gridded_${radar_name}_${date_str}${time_str}.nc -or cfradial_${radar_name}_${date_str}${time_str}.nc
done

date
