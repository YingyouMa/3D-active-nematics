"""
Simulation Script of 3D active nematics on LAMMPS
Arthor: Yingyou Ma, Physics @ Brandeis, 2024, March

This script automates the setup and execution of 3D active nematics simulations on LAMMPS with varying parameters.
It creates simulation directories, makes the initial configuration, generates input files, and submits jobs to a cluster using SLURM.
The simulations are designed with 3D active semi-flexible filaments.

Dependencies:
- LAMMPS (with our active code embedded)
- Python 3.x
- NumPy: 1.22.0
- Pandas

Instructions:
1. Ensure LAMMPS is installed with active code.
2. Adjust constants and path as needed. The path of each simulation is defined in pathname.py.
3. Run the script to start LAMMPS simulations.

Note: Make sure to customize the script for your specific simulation setup.
"""

# Built-in packages
from pathlib import Path
from subprocess import call, check_output
from io import StringIO
import os
import glob
import re

# External packages
import pandas as pd
import numpy as np

# Custom packages
from pathname import get_mainpath

# Directory
ROOT = Path(__file__).resolve().parent
CREATE_POLY_PATH = ROOT / "tools/create_poly.py" # Path to the create_poly.py script, which generates the initial condition.
PARAMETERS_PATH = ROOT / "parameters.csv"

# Constants of filaments
WIDTH               = 200       # Box width
DENSITY             = 0.7       # Volume fraction
NATOMS              = 50        # Amount of monomers per filament
DAMP                = 0.1       # Damping coefficient (one over gamma)
BOND_LENGTH         = 0.5       # Equilibrium length of bond

# Constants of simulation
ARCHIVE_INTV_RAW    = 800_000   # Raw interval of storing the restart files. The real interval might be slightly changed
DUMP_BASE           = 200       # Set the dumping intervals as multipole of DUMP_BASE
TIME_STEP           = 0.001     # Timestep for the simulation


# Loading templates for LAMMPS and SLURM
with open("templates/lammps.template", "r") as f:
    lammps_template = f.read()

with open("templates/lammps.restart.template", "r") as f:
    restart_template = f.read()

with open("templates/sbatch.template", "r") as f:
    sbatch_template = f.read()

# Check the current status of simulation
check_command = "squeue --me -o %i,%j,%t,%S"
check_out = check_output( check_command.split() ).decode('utf8')
check_table = pd.read_csv(StringIO(check_out))
print(check_table)


# Read the parameters for simulations from parameters.csv
parameters = np.array(pd.read_csv(PARAMETERS_PATH))

# The directory's name of upcoming simulations
name_list = np.arange(1, 4)

# The function to run the simulation of each set of parameters
def main(parameter, name):

    # Load the parameters and calculate the dump interval
    stiffness, activity, max_time, dump_N = parameter
    stiffness = int(stiffness)
    activity  = float(activity)
    max_time  = int(max_time)
    dump_N    = int(dump_N)
    max_step  = max_time/TIME_STEP
    dump_intv = max_step/dump_N
    if dump_intv > ARCHIVE_INTV_RAW:
        dump_intv = ARCHIVE_INTV_RAW
    dump_intv = round(dump_intv / DUMP_BASE) * DUMP_BASE
    max_step  = round(max_step / dump_intv) * dump_intv
    if max_step * TIME_STEP >= max_time:
        max_time  = max_step * TIME_STEP
    else:
        max_step = max_step + dump_intv
        max_time = max_step * TIME_STEP
    archive_intv = round( ARCHIVE_INTV_RAW / dump_intv) * dump_intv
    if archive_intv > max_step:
        archive_intv = max_step

    job_name = f"Nematics3D_k{stiffness}_a{activity}_n{name}"
    path = get_mainpath(DENSITY, stiffness, activity, name)


    print(f"Checking job {job_name}")

    if job_name in check_table['NAME'].values:
        print(f" Already running")
        return 0

    path.mkdir(exist_ok=True, parents=True)
    os.chdir(path)

    # check if the simulation has finished
    # Firstly, check if the compeletion flag is created
    if os.path.isfile( path / 'end.txt' ):
        with open(path / 'end.txt', 'r') as f:
            end = float(f.readline().strip())
        if end >= max_time:
            print('  The simulation has finished')
            return 0
    
    # Next, check the dump directory
    # If the simulation time of the latest dump file is longer than set, create the completetion flag
    dump_files = glob.glob('dump/*data*')
    frames = np.array([int(re.findall(r'\d+', file)[-1]) for file in dump_files])
    if len(frames)>0:
        dump_final = np.sort(frames)[-1]
        if dump_final >= max_step:
            with open('end.txt', 'w') as f:
                f.write(f'{dump_final*TIME_STEP}')
            print('  The simulation has finished')
            return 0

    # Check if it's a new simulation
    save_version = 1
    if Path("restart/save.base.dat.1").is_file():
        print("... Found archived data. Continuing simulation...")
        lammps_script = "restart.lmp"
        sim_template = restart_template
        
        try:
            mtime_v1 = os.path.getmtime("restart/save.base.dat.1")
            mtime_v2 = os.path.getmtime("restart/save.base.dat.2")
            save_version = 1 if mtime_v1 > mtime_v2 else 2
        except FileNotFoundError:
            save_version = 1
    else:
        print("... Not found archived data. Starting new simulation...")
        lammps_script = "in.sim.lmp"
        sim_template = lammps_template

    # Create directories
    os.makedirs("./dump", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./restart", exist_ok=True)

    # Create the lammps file
    with open(lammps_script, "w") as f:
        contents = sim_template.format(
            width=WIDTH,
            cutoff=2**(1/6),
            seed=np.random.randint(1000000),
            stiffness=stiffness,
            activity=activity/DAMP,
            damp=DAMP,
            tau=1,
            dump_intv=dump_intv,
            archive_intv=archive_intv,
            save_version=save_version,
            dump_path="dump/*.data",
            timestep=TIME_STEP,
            max_step=max_step,
            max_time=max_time
        )
        f.write(contents)  

    # Create the sbatch file and the function creating the initial filaments
    create_poly_cmd = f"python3 {CREATE_POLY_PATH} -w {WIDTH} -n {NATOMS} -v {DENSITY} -b {BOND_LENGTH}"
    with open("submit.sh", "w") as f:
        contents = sbatch_template.format(
            job_name=job_name,
            create_poly_cmd=create_poly_cmd,
            lammps_script=lammps_script
        )
        f.write(contents) 

    # RUN!
    call("sbatch submit.sh".split())

    os.chdir(ROOT) 

# Run each simulation
for name in name_list:
    for parameter in parameters:
        main(parameter, name)
