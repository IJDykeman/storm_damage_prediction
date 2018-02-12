'''
This file just creates a model in verbose mode, printing out
relevant information as defined in Model.
'''

import model
import time

t1 = time.time()
model = model.Model(verbose = True)
model.restore()
print "initializing and restoring the model takes", time.time()-t1, "seconds"
