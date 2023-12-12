#! /usr/bin/env python3

import argparse
import os
from io import StringIO
from pathlib import Path
from itertools import product
from subprocess import call, check_output
import random

import numpy as np
import pandas as pd

# CONSTANTS
ROOT = Path(__file__).resolve().parent
BOND_LENGTH = 0.50

DUMP_FREQ       =  1_000
ARCHIVE_FREQ    =  1_000_000_000   
MAX_STEP        =  3_200_000

CREATE_POLY_PATH = ROOT / "tools/create_poly"

# PARAMETERS
WIDTH = [200]

DENSITY = [0.7]
NATOMS = [50]
STIFFNESS = [100]
ACTIVITY = [1]
SEED = [1000]

with open("templates/lammps.template", "r") as f:
    lammps_template = f.read()

with open("templates/lammps.restart.template", "r") as f:
    restart_template = f.read()

with open("templates/sbatch.template", "r") as f:
    sbatch_template = f.read()

out = check_output("squeue -u yingyou -o %i,%j,%t,%S".split()).decode('utf8')
table = pd.read_csv(StringIO(out))

print(table)

d = 0.1
# w = 200
for (w, rho, n, k, a, s) in product(WIDTH, DENSITY, NATOMS, STIFFNESS, ACTIVITY, SEED):
    job_name = f"begin_W{w}_D{rho}_N{n}_K{k}_A{a}_d{d}_s{s}"
    path = ROOT / f"data/width_{w}/viscosity_{int(1/d)}/density_{rho:0.2f}/length_{n}/stiffness_{k}/activity_{a}/seed_{s}"

    print(f"Checking job {job_name}")

    if job_name in table['NAME'].values:
        print(f"  Job '{job_name}' already running")
        continue
    
    path.mkdir(exist_ok=True, parents=True)
    os.chdir(path)

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

    os.makedirs("./dump", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)
    os.makedirs("./restart", exist_ok=True)

    with open(lammps_script, "w") as f:
        contents = sim_template.format(
            width=w,
            cutoff=2**(1/6),
            seed=s,
            stiffness=k,
            activity=int(a/d),
            damp=d,
            tau=1,
            dump_freq=DUMP_FREQ,
            archive_freq=ARCHIVE_FREQ,
            save_version=save_version,
            dump_path="dump/*.mpiio.data",
            timestep=1e-3,
            max_step=MAX_STEP,
        )
        f.write(contents)
    
    create_poly_cmd = f"python3 {CREATE_POLY_PATH} -w {w} -n {n} -v {rho} -b {BOND_LENGTH}"
    with open("submit.sh", "w") as f:
        contents = sbatch_template.format(
            job_name=job_name,
            create_poly_cmd=create_poly_cmd,
            lammps_script=lammps_script
        )
        f.write(contents)

    call("sbatch submit.sh".split())

    os.chdir(ROOT)
