

# this will not work becuase the binary format of this data cannot be read by open souce tools.

import sys
from osgeo import ogr
ogr.UseExceptions()

driver = ogr.GetDriverByName("OpenFileGDB")
print "driver:", driver
try:
    gdb = driver.Open("./HGAC_Land_Cover_10_Class_2008", 0)
except Exception as e:
    print e
    sys.exit()

for featsClass_idx in range(gdb.GetLayersCount()):
    featsClass = gdb.GetLayerByIndex(featsClass_idx)
    print featsClass.GetName()
