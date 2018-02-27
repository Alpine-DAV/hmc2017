import csv
import sys

class CSVOutput(object):
    def __init__(self, schema, write_schema=False, output=None, append=False):
        if output is None:
            self.f_ = sys.stdout
        else:
            self.f_ = open(output, 'ab' if append  else 'wb')

        self.schema_ = schema
        self.writer_ = csv.writer(self.f_)

        if write_schema:
            self.writer_.writerow(self.schema_)

    def writerow(self, **cols):
        if sorted(cols.keys()) != sorted(self.schema_):
            raise ValueError('columns in row do not match schema (got {}, expected {})'.format(
                sorted(cols.keys()), sorted(self.schema_)))
        self.writer_.writerow([cols[col] for col in self.schema_])

    def close(self):
        if self.f_ is not sys.stdout:
            self.f_.close()
