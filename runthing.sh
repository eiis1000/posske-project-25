echo "This is my test bash script"
echo "Activating conda environment"
eval "$('/home/erezm/miniconda3/bin/conda' 'shell.bash' 'hook')"
conda activate sage
echo "Running python script"
python deformations/compute_lrid.py