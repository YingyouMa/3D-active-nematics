from pathlib import Path

ROOT = Path(__file__).resolve().parent

def get_mainpath(density, stiffness, activity, name):
    return ROOT / f"data/density_{density:0.2f}/stiffness_{stiffness}/activity_{activity}/{name}"

def get_coarsepath(density, stiffness, activity, name):
    return ROOT / f"coarse/density_{density:0.2f}/stiffness_{stiffness}/activity_{activity}/{name}"

def get_diagpath(density, stiffness, activity, name):
    return ROOT / f"diag/density_{density:0.2f}/stiffness_{stiffness}/activity_{activity}/{name}/"