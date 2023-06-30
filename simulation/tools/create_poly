#! /usr/bin/env python

import argparse
import json
import sys

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("-w", "--width",
                    help="The domain box width",
                    type=float,
                    dest="width",
                    default=50)

parser.add_argument("-n", "--atoms-per-polymer",
                    help="The number of particles in the polymer",
                    type=int,
                    dest="N",
                    default=20)

parser.add_argument("-v", "--volume-fraction",
                    help="The volume fraction",
                    type=float,
                    dest="vf",
                    default=0.65)

parser.add_argument("-b", "--bond_length",
                    help="the mean bond length of each polymer",
                    type=float,
                    dest="b",
                    default=0.5)

args = parser.parse_args()

# now that all of the arguments are parsed, we grab the relevant variables
W = args.width
vf = args.vf
N = args.N
b = args.b

if vf >= 1.0 or vf <= 0.0:
    raise ValueError(
        "Volume fraction must be in range (0, 1). Value requested = {}"
        .format(vf))

# to get the polymer length, note that there are N-1 bonds with mean bond length b
# additionally, the beads at the end each contribute a bead radius to the length
L = (N - 1)*b + 1

# determine number of polymers
M = vf * W**3 / ((N - 1)*b*np.pi*(0.5**2) + (4.0*np.pi/3.0)*(0.5**3))
M = int(round(M))

# naive volume fraction and real volume fraction
vol_frac_naive = M*N*(4.0/3.0)*np.pi*(0.5**3) / (W**3)
vf = M * ((N - 1)*b*np.pi*(0.5**2) + (4.0*np.pi/3.0)*(0.5**3)) / (W**3)

num_atoms = M*N

# write all of this to a json file for later use
parameters = {
    "atoms_per_polymer": N,
    "num_polymers": M,
    "num_atoms": num_atoms,
    "bond_length": b,
    "polymer_length": L,
    "system_width": W,
    "vol_frac_naive": vol_frac_naive,
    "vol_frac": vf,
    "confinement": L/W
}

print("Saving to JSON file...")
with open("parameters.json", "w") as f:
    json.dump(parameters, f, indent=4)
print("Success!")


print("Writing polymer data file")
with open("atoms.txt", "w") as f:
    f.write("Number of polymers = {} of size {}\n\n".format(M, N))

    f.write("{} atoms\n".format(N*M))
    f.write("{} bonds\n".format((N-1)*M))
    f.write("{} angles\n".format((N-2)*M))
    f.write(f"{int(-0.5*W)} {int(1.5*W)} xlo xhi\n")
    f.write("0 {} ylo yhi\n".format(W))
    f.write("0 {} zlo zhi\n".format(W))
    f.write("{} atom types\n".format(1))
    f.write("{} bond types\n".format(1))
    f.write("{} angle types\n".format(1))

    f.write("\nAtoms # angle \n\n")
    # f.write("# atom_id atom_type x y z mol_id diameter density\n")
    for i in range(num_atoms):
        # define particle positions
        if i % N == 0:
            x0 = W * np.random.rand()
            y = W * np.random.rand()
            z = W * np.random.rand()
        x = x0 + b * (i % N)

        atom_id = i + 1
        mol_id = (i // N) + 1
        atom_type = 1
        diam = 1.0
        density = 1.0

        f.write("{} {} {} {:0.2f} {:0.2f} {:0.2f}\n".format(
            atom_id, mol_id, atom_type, x, y, z))

    f.write("\nBonds\n\n")
    bond_id = 1
    bond_type = 1
    for i in range(num_atoms):
        atom_id = i + 1
        idx = (i % N) + 1
        if idx + 1 > N:
            continue
        f.write("{} {} {} {}\n".format(
            bond_id, bond_type, atom_id, atom_id + 1))
        bond_id += 1

    f.write("\nAngles\n\n")
    angle_id = 1
    angle_type = 1
    for i in range(num_atoms):
        atom_id = i + 1
        idx = (i % N) + 1
        if idx + 2 > N:
            continue
        f.write("{} {} {} {} {}\n".format(
            angle_id, angle_type, atom_id, atom_id + 1, atom_id + 2))
        angle_id += 1
