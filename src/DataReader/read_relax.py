import sys, os
import numpy as np
from FeatureDataReader import FeatureDataReader


if len(sys.argv) < 2:
    print 'Usage: read_relax.py dataDir cycle'
    sys.exit(1)

np.set_printoptions(suppress=True, precision=4)

reader = FeatureDataReader(sys.argv[1])

names = reader.getFeatureNames()
print names

try:
    index = names.index('relaxDistance')
except:
    print 'Error: no relaxation distance in feature data'
    sys.exit(1)
    
zids = reader.getCycleZoneIds()
data = reader.readAllZonesInCycle(0, int(sys.argv[2]))

relax = data[:,index]
zones = np.where(relax > 0)

rids = zids[zones[0]]
vals = data[zones[0],:]

for n,zid in enumerate(rids):
    print 'Zone %d:' % zid
    print vals[n]
