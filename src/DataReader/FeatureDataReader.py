import numpy as np
import sys, os, re


class FeatureDataReader(object):
    """Reader for simulation data related to machine learning features
    
    Attributes:
        dataDir: data directory
        numParts: # of partitions in mesh
        zoneOffsets: a map that takes zone id and returns partition and
                      offset within
        indexCache: cache content of index file, which contains starting
                     position (for seek) of each cycle
        metaData: cache content of meta data file, which contains features
                   names and zone ids
    """

    def __init__(self, dataDir):
        """Class constructor
        
        Args:
            dataDir: data directory
            numParts: # of partitions in mesh
        """
        self.dataDir = dataDir

        # read numParts from simulationInfo.txt
        with open("%s/simulationInfo.txt" % self.dataDir, 'r') as fin:
            numProcs = fin.readline().rstrip('\n').split(':')
            self.numParts = int(numProcs[1])

        # save content to avoid recomputing them
        self.zoneOffsets = {}
        self.indexCache = {}
        self.metaData = {}
        

    def readZone(self, run, cycle, zone):
        """Read data from a single mesh zone from a simulation cycle of a run
        
        Args:
            run: simulation run #
            cycle: simulation cycle # (time step)
            zone: mesh zone id
        
        Returns:
            1D numpy array of size (# of features)
        """
        zid = self.getZoneOffset(zone)
        index = self.readFileIndex(run, zid['part'])
        meta = self.readMetaData(zid['part'])
        
        fname = 'features_p%03d_r%03d.npy' % (zid['part'], run)
        numFeats = len(meta['features'])
        offset = zid['offset'] * numFeats * 4

        with open("%s/features/%s" % (self.dataDir,fname), 'rb') as fin:
            fin.seek(index[cycle] + offset)
            return np.fromfile(fin, dtype=np.float32, count=numFeats)


    def readPartition(self, run, part, cycle):
        """Read data from entire mesh partition from a simulation cycle of a run
        
        Args:
            run: simulation run #
            part: mesh partition #
            cycle: simulation cycle # (time step)
            
        Returns:
            2D numpy array of shape (# of zones in partition X # of features)
        """
        index = self.readFileIndex(run, part)
        meta = self.readMetaData(part)

        fname = 'features_p%03d_r%03d.npy' % (part, run)
        numFeats = len(meta['features'])
        numZones = len(meta['zones'])
        
        with open("%s/features/%s" % (self.dataDir,fname), 'rb') as fin:
            fin.seek(index[cycle])
            data = np.fromfile(fin, dtype=np.float32, count=numZones*numFeats)
            return np.reshape(data, (numZones,numFeats))


    def readAllCyclesForZone(self, run, zone):
        """Read data from all simulation cycles in a run of a single mesh zone
        
        Args:
            run: simulation run #
            zone: mesh zone id
            
        Returns:
            2D numpy array of shape (# of cycles X # of features)
        """
        zid = self.getZoneOffset(zone)
        index = self.readFileIndex(run, zid['part'])
        meta = self.readMetaData(zid['part'])
        
        fname = 'features_p%03d_r%03d.npy' % (zid['part'], run)
        numFeats = len(meta['features'])
        offset = zid['offset'] * numFeats * 4

        with open("%s/features/%s" % (self.dataDir,fname), 'rb') as fin:
            data = []
            for cycle in sorted(index.keys()):
                fin.seek(index[cycle] + offset)
                single = np.fromfile(fin, dtype=np.float32, count=numFeats)
                data.append(np.reshape(single, (1,numFeats)))
            return np.concatenate(data)


    def readAllZonesInCycle(self, run, cycle):
        """Read data from all mesh zones from a simulation cycle of a run
        
        Args:
            run: simulation run #
            cycle: simulation cycle # (time step)

        Returns:
            2D numpy array of shape (# of zones in mesh X # of features)
        """
        data = []
        for part in range(0, self.numParts):
            data.append(self.readPartition(run, part, cycle))
        return np.concatenate(data)


    def getPartitionZoneIds(self, part):
        """Get zone ids for a mesh partition
        
        Must ensure zone ids are in same order as read in from meta data file
        
        Args:
            part: mesh partition #
            
        Returns:
            1D numpy array of size (# of zones in partition)
        """
        zones = self.readMetaData(part)['zones']
        ids = [id for id in sorted(zones, key=zones.get)]
        return np.asarray(ids, dtype=np.int32)


    def getCycleZoneIds(self):
        """Get zone ids for entire mesh
        
        Returns:
            1D numpy array of size (# of zones in mesh)
        """
        data = []
        for part in range(0, self.numParts):
            data.append(self.getPartitionZoneIds(part))
        return np.concatenate(data)


    def getFeatureNames(self):
        """Get names (or labels) of feature features
        
        Must ensure names are in same order as read in from meta data file
        All mesh partitions have the same set of features
        
        Returns:
            list of size (# of features)
        """
        features = self.readMetaData(0)['features']
        return [name for name in sorted(features, key=features.get)]


    def readFileIndex(self, run, part):
        """Read file index and cache content into dictionary for mesh partition
        
        File index contains the seek position for the start of each cycle
        This speeds up the process for reading data from each cycle
        For each mesh partition, a separate dictionary is constructed that can 
        be indexed by the partition file name
        Essentially, this is a dictionary of dictionaries
        
        Args:
            run: simulation run #
            part: mesh partition #

        Returns:
            Dictionary with keys as cycle # and values as seek position
        """
        fname = 'indexes_p%03d_r%03d.txt' % (part, run)
        if fname not in self.indexCache:
            with open("%s/indexes/%s" % (self.dataDir,fname), 'r') as fin:
                index = {}
                for line in fin:
                    key,val = line.split(' -> ')
                    index[int(key)] = int(val)
                self.indexCache[fname] = index
        return self.indexCache[fname]


    def readMetaData(self, part):
        """Read meta data and cache content into dictinary for mesh partition
        
        Meta data file contains feature feature names and mesh zone ids
        For each mesh partition, a separate dictionary is constructed that can 
        be indexed by the partition file name
        Each dictionary contains two elements: feature names and zone ids, both 
        of which are dictionaries themselves:
            Features is a dictionary with keys as names and values as offsets
            Zones is a dictionary with keys as ids and values as offsets
        The purpose of these two dictionaries is to enable users to find which 
        row or column within the 2D numpy array corresponds to which zone or 
        feature value, respectively
        Essentially, this is a dictionary of dictionaries of dictionaries
        
        Args:
            part: mesh partition #
            
        Returns:
            Dictionary of two dictionaries: feature names and zone ids
        """
        fname = 'metadata_p%03d.txt' % part
        if fname not in self.metaData:
            with open("%s/features/%s" % (self.dataDir,fname), 'r') as fin:
                
                fin.readline()  # skip features header
                # strip away newline and extra comma
                names = fin.readline().rstrip('\n,').split(',')
                features = dict(zip(names, range(0, len(names))))

                fin.readline()  # skip zones header
                ids = fin.read().splitlines()
                zones = dict(zip(map(int, ids), range(0, len(ids))))

                self.metaData[fname] = {'features': features, 'zones': zones}
        return self.metaData[fname]


    def getZoneOffset(self, zone):
        """Get mesh partition # and offset within partition for a single zone
        
        Construct a dictionary using zone ids as keys and partition plus offset 
        as values
        This requires first reading (and caching) the meta data files for all 
        mesh partitions
        
        Args:
            zone: mesh zone id
            
        Returns:
            Dictionary with keys as zone ids and values as partition and offset
        """
        if not self.zoneOffsets:
            for part in range(0, self.numParts):
                meta = self.readMetaData(part)
                for key in meta['zones'].keys():
                    self.zoneOffsets[key] = {'part': part,
											 'offset': meta['zones'][key]}
        return self.zoneOffsets[zone]


    def getRunNumbers(self):
        """Get the simulation run numbers from timing.txt file

		Look for the pattern: Timing for run###

        Returns:
            List of run numbers as ints
        """
        nums = []
        with open('%s/timing.txt' % self.dataDir, 'r') as fin:
            for line in fin:
                num = re.findall('Timing for run(\d+)', line)
                if len(num) == 1:
					nums.append(int(num[0]))
        return nums


    def getAllFailures(self):
        """Get the run #, cycle #, and zone id for each failure

        The failures/ directory contains text files for both types of failures:
        negative side or corner volume, divided into partitions
        e.g., side_p001 contains all the negative side volume failures from
        partition 1

        Returns:
            List of tuples (run,cycle,zone) for each failure, sorted by run #
        """
        failures = []
        files = os.listdir('%s/failures' % self.dataDir)

        for file in files:
            with open('%s/failures/%s' % (self.dataDir,file), 'r') as fin:
                state = 0
                for line in fin:
                    # strip away newline and extra comma
                    vals = line.rstrip('\n,').split(',')
                    
                    if vals[0] == 'Run':
                        state = 1
                    elif vals[0] == 'volume':
                        state = 2
                    else:
                        if state == 1:
                            failures.append(map(int, vals))
                        else:
                            failures[-1].append(map(float, vals))
                            
        return sorted(failures, key=lambda x: x[0])


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Usage: FeatureDataReader.py dataDir'
        sys.exit(1)

    np.set_printoptions(suppress=True, precision=4)

    reader = FeatureDataReader(sys.argv[1])
    (run, part, cycle, zone) = (0, 0, 0, 0)

    data = reader.getFeatureNames()
    print 'Features:', data

    data = reader.readZone(run, cycle, zone)
    print 'Zone:', data

    data = reader.readPartition(run, part, cycle)
    print 'Partition:', data.shape

    data = reader.readAllZonesInCycle(run, cycle)
    print 'AllZonesInCycle:', data.shape

    data = reader.readAllCyclesForZone(run, zone)
    print 'AllCyclesForZone:', data.shape

    data = reader.getPartitionZoneIds(part)
    print 'PartitionZoneIds:', data.shape
    
    data = reader.getCycleZoneIds()
    print 'CycleZoneIds:', data.shape

    data = reader.getRunNumbers()
    print 'Runs: %d, %d, ..., %d' % (data[0], data[1], data[-1])
    
    data = reader.getAllFailures()
    print 'Failure:', data[0]
