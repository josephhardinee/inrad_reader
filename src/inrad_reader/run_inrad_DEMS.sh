#!/bin/bash
#SBATCH -A m1657
#SBATCH --job-name=run_inrad.py
#SBATCH -p regular
#SBATCH --ntasks=1
#SBATCH -C haswell
#SBATCH --mail-user=jingyi.chen@pnnl.gov
#SBATCH --nodes=1
#SBATCH --time=00:10:00
#SBATCH --output=./log/run_inrad_DEMS.out
#SBATCH --exclusive

# enter the folder
cd /global/homes/c/chen696/inrad_reader/inrad_reader/src/inrad_reader

# python environment
bash
source /global/common/cori/software/python/3.6-anaconda-5.2/bin/activate /global/homes/c/chen696/.conda/envs/myenv


#  start loop through selected files
date
radar_name="DEMS"
date_str="20180828"

#for time_str in "005229" "022230" "035230" "060230" "080229" "090024" "113230" "142023" "144229"
for time_str in "005229" "022230"
do

	input_name="/project/projectdirs/m1657/zfeng/indian_radar/${radar_name}-RADAR/${date_str}/T_HAHA00_C_${radar_name}_${date_str}${time_str}"
	echo $input_name\*
	python inrad_to_cf.py -fg $input_name\* -c example_config.json -og gridded_${radar_name}_${date_str}${time_str}.nc -or cfradial_${radar_name}_${date_str}${time_str}.nc
done

date
