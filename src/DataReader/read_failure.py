import sys, os
import numpy as np
from FeatureDataReader import FeatureDataReader


if len(sys.argv) < 2:
    print 'Usage: read_failure.py dataDir'
    sys.exit(1)

np.set_printoptions(suppress=True, precision=4)

reader = FeatureDataReader(sys.argv[1])
fails = reader.getAllFailures()

meta = reader.readMetaData(0)
numFeats = len(meta['features'])

for i in range(len(fails)):
    (run,cycle,zone,vals) = fails[i]

    with open('%s/timeSeries/failure_r%04d_z%06d.npy' % 
              (sys.argv[1],run,zone), 'rb') as fin:
        # assumes contiguous cycles starting at 0
        # failure cycle is the last cycle of that time series
        data = np.fromfile(fin, dtype=np.float32, count=(cycle+1)*numFeats)
        data = np.reshape(data, (cycle+1,numFeats))
        
        print 'Failure-%d:' % i
        print np.asarray(vals)
        print data[cycle]
        