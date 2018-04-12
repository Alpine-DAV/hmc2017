# HMC17-18 LLNL Clinic Repository

This is the repository for the 2017-2018 LLNL Harvey Mudd Clinic Project. Its purpose is to investigate the effect of parallel distributed-memory machine learning, and to construct a functional model that will work in LLNL's in-situ simulation runs. Our final implementation uses Mondrian Forests with modified serialization in the scikit-garden implementation to speed up model reductions (the process of combining local models into one global model).

## Running the code

1. Install Python 2.7:
  * https://conda.io/docs/user-guide/install/download.html

2. Download the following fork of scikit-garden, featuring a Mondrian Forest implementation:
```
   git clone https://github.com/jbearer/scikit-garden.git
   cd scikit-garden
   python setup.py install
```

3. Install MPI4Py:
https://mpi4py.readthedocs.io/en/stable/install.html#using-pip-or-easy-install
`pip install mpi4py`

4. Run the following to see if everything is functional 
`mpiexec -n 8 python mpiml/bin/learning_wrapper.py [optionals] path/to/{bubbleShock, bubbleShock_byHand} {nb, rf, mf}`

## Contributors
Katelyn Barnes, Jeb Bearer, Evan Chrisinger and Amy Huang 