#! /bin/bash
#SBATCH -A TG-MCB090163
#SBATCH -p compute
#SBATCH -J {job_name}
#SBATCH -o logs/output.txt
#SBATCH -e logs/error.txt
#SBATCH -N 2
#SBATCH --ntasks-per-node 128
#SBATCH -t 48:00:00
#SBATCH --mail-user=yingyouma@brandeis.edu
#SBATCH --mail-type=end
#SBATCH --mem=249208M

# Load openmpi. 
# The code could be different on different server.
module load cpu/0.17.3b
module load gcc/10.2.0/npcyll4
module load openmpi/4.1.1

# If this is a new simulation which lacks data of initial condition (atoms.txt), create it.
if [ ! -f "atoms.txt" ] || [ $(ls -1 restart/ | wc -l) -eq 0 ];
then
    {create_poly_cmd}
fi

# Run lammps. 
# The command could be different on different server.
# The first argument is the address of excutable lammps program
srun -n $SLURM_NTASKS /home/yingyou/lammps/build/lmp -in {lammps_script}
