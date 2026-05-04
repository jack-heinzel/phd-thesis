"""Find acronym definitions that already exist in the common files.

Note - this is not for tests in the common files, but to be used to
supermodules that are using the common files
"""

import sys
import re


local_file = sys.argv[1]
common_file = sys.argv[2]


def read_acro_defs(acro_def_path):
    with open(acro_def_path, 'r') as acro_def_file:
        lines = acro_def_file.readlines()
    acro_defs = set()
    for line in lines:
        match = re.search(r'\\acrodef{([A-Za-z0-9]+)}', line)
        if match:
            acro_defs.add(match.group(1))
            continue
    return acro_defs


local_acro_defs = read_acro_defs(local_file)
common_acro_defs = read_acro_defs(common_file)

dupe_acro_defs = local_acro_defs & common_acro_defs

if dupe_acro_defs:
    print('---- Duplicate acronym definitions found ----')
    for ad in sorted(dupe_acro_defs):
        print(ad)
    sys.exit(1)
