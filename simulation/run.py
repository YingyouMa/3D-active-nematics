from pathlib import Path
from subprocess import call, check_output
from io import StringIO
import os

import pandas as pd
import numpy as np

import datetime

# Directory
ROOT = Path(__file__).resolve().parent
CREATE_POLY_PATH = ROOT / "tools/create_poly"

# Constants of filaments
WIDTH           = 200       # Box width
DENSITY         = 0.7       # Volume fraction
NATOMS          = 50        # Amount of monomers per filament
DAMP            = 10        # Damping coefficient (one over gamma)
BOND_LENGTH     = 0.5       # Equilibrium length of bond

# Constants of simulation
DUMP_N          = 200       # Total number of dumping files
ARCHIVE_INTV    = 120_000   # Interval of storing the restart files
DUMP_BASE       = 500       # Set the dumping intervals as multipole of DUMP_BASE
TIME_STEP       = 0.001


# Loading templates
with open("templates/lammps.template", "r") as f:
    lammps_template = f.read()

with open("templates/lammps.restart.template", "r") as f:
    restart_template = f.read()

with open("templates/sbatch.template", "r") as f:
    sbatch_template = f.read()

# Check the current status of simulation
check_command = "squeue -u yingyou -o %i,%j,%t,%S" # Needs to be changed for different servers
check_out = check_output( check_command.split() ).decode('utf8')
check_table = pd.read_csv(StringIO(check_out))
print(check_table)


# Read the parameters for simulations
parameters = np.array(pd.read_csv("parameters.csv"))

# The directory's name of upcoming simulations
name_list = np.arange(5)

# The function to run the simulation of each set of parameters
def main(parameter, name):

    # Load the parameters and calculate the dump interval
    stiffness, activity, max_time = parameter
    stiffness =int(stiffness)
    dump_intv = max_time/TIME_STEP/DUMP_N
    dump_intv = round(dump_intv / DUMP_BASE) * DUMP_BASE
    max_step  = int( dump_intv * DUMP_N )

    job_name = f"Nematics3D_k{stiffness}_a{activity}_n{name}"
    path = ROOT / f"data/density_{DENSITY:0.2f}/stiffness_{stiffness}/activity_{activity}/{name}"

    print(f"Checking job {job_name}")

    if job_name in check_table['NAME'].values:
        print(f"  Job '{job_name}' already running")
        return 0

    path.mkdir(exist_ok=True, parents=True)
    os.chdir(path)

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
        print("...  Starting new simulation...")
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
            activity=int(activity/DAMP),
            damp=DAMP,
            tau=1,
            dump_intv=dump_intv,
            archive_intv=ARCHIVE_INTV,
            save_version=save_version,
            dump_path="dump/*.mpiio.data",
            timestep=TIME_STEP,
            max_step=max_step,
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
