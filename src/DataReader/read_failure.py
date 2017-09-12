import sys, os
import numpy as np
from FeatureDataReader import FeatureDataReader


if len(sys.argv) < 2:
    print 'Usage: read_failure.py dataDir'
    sys.exit(1)

np.set_printoptions(suppress=True, precision=4)

reader = FeatureDataReader(sys.argv[1])
fails = reader.getAllFailures()

for i in range(len(fails)):
    (run,cycle,zone,vals) = fails[i]
    data = reader.readAllCyclesForFailedZone(run,cycle,zone)
        
    print 'Failure-%d:' % i
    print np.asarray(vals)
    print data[cycle]
        
