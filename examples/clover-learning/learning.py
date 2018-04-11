# Example using mpiml's Mondrian forest implementation to predict physical quantities in a physics
# simulation. The particular learning problem demonstrated is simple: use energy to predict
# pressure. The learning phase runs every 10 timesteps and lasts 250 timesteps total. After that,
# the script uses the model it has trained to start predicting.
#
# For each prediction it makes, the script uses Ascent inception to render images of the predicted
# quantity, the true value of the quantity, and the difference between the two.
#
# This script is designed to run alongside the Clover proxy simulation using an Ascent extract. The
# accompanying file ascent_actions.json points Ascent to this script, and the file clover.in
# convfigures Clover with initial conditions and runtime parameters.

import ascent.mpi
import atexit
import conduit
from mpi4py import MPI
from mpiml.forest import MondrianForestRegressor
import numpy as np
import os
import time

def mv(src, dest):
    if MPI.COMM_WORLD.rank == 0:
        os.rename(src, dest)
    MPI.COMM_WORLD.barrier()

# We have to rename the ascent_actions.json file, or else the nested Ascent instance will pick it up
# and call this script recursively, rather than rendering the scene we want it to render.
mv('ascent_actions.json', '_ascent_actions.json')
atexit.register(mv, '_ascent_actions.json', 'ascent_actions.json') # Move it back when we're done

# Hack (idiom?) to initialize prd once and have it persist through each iteration
if 'prd' not in globals():
    prd = MondrianForestRegressor()

data = ascent_data()
cycle = data["state"]["cycle"]

# Learning problem: x, an array with a single feature (energy), and y, an array of pressure values
x = data["fields"]["energy"]["values"].reshape(-1, 1)
y = data["fields"]["density"]["values"]

if cycle < 250:
    prd.partial_fit(x, y)
else:
    out = prd.reduce().predict(x)

    # Hack to initialize ml_output and ml_diff with sane metadata
    data['fields/ml_output'] = data['fields/pressure']
    data['fields/ml_diff'] = data['fields/pressure']

    # Annotate the mesh with the results of our prediction
    data['fields/ml_output/values'] = out
    data['fields/ml_diff/values'] = out - y

    # Create a nested instance of ascent and configure it to render some pictures
    a = ascent.mpi.Ascent()

    ascent_opts = conduit.Node()
    ascent_opts['mpi_comm'].set(MPI.COMM_WORLD.py2f())
    a.open(ascent_opts)

    # Send the annotated mesh to Ascent
    a.publish(data)

    scenes = conduit.Node()

    # Plot actual pressure
    scenes['s1/plots/p1/type'] = 'pseudocolor'
    scenes['s1/plots/p1/params/field'] = 'pressure'
    scenes['s1/image_prefix'] = 'pressure'

    # Plot predicted pressure
    scenes['s2/plots/p1/type'] = 'pseudocolor'
    scenes['s2/plots/p1/params/field'] = 'ml_output'
    scenes['s2/image_prefix'] = 'ml_output'

    # Plot difference between actual and predicted pressure
    scenes['s3/plots/p1/type'] = 'pseudocolor'
    scenes['s3/plots/p1/params/field'] = 'ml_diff'
    scenes['s3/image_prefix'] = 'ml_diff'

    actions = conduit.Node()
    add_act = actions.append()
    add_act['action'] = 'add_scenes'
    add_act['scenes'] = scenes

    actions.append()['action'] = 'execute'

    a.execute(actions)
    a.close()
