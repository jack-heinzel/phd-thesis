"""
First, run this script on a subset of bib keys
that are defined in inspirehep. Then pass this to
create_bibliography.py in gwtc-common-files
"""

import sys
import numpy as np

fname = sys.argv[1]

list_of_chars = []
list_of_refs = []
with open(fname, 'r') as f:
    for line in f:
        if line.startswith('@'):
        	split_list = line[1:].split("{")

        	list_of_chars.append(split_list[0])
        	list_of_refs.append(split_list[1][:-2])

print(np.unique(list_of_chars))
list_of_refs = np.array(list_of_refs, dtype=str)
print(list_of_refs)
np.savetxt("bibliography.keys", list_of_refs, fmt="%s")
