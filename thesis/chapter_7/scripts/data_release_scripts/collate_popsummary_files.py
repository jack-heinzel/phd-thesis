import filecmp
import sys
import glob
import os
import numpy as np
import shutil

DIR = "data_release"
FILELIST = glob.glob("../../analyses/*/*txt")
for FNAME in FILELIST:
    # print(FNAME)
    BASENAME = os.path.basename(FNAME)
    analysis_label = BASENAME.split("_")[2]
    # Check if the words "BBH" "NS" "CBC" are in the analysis_label
    if not any(word in analysis_label for word in ["BBH", "NS", "CBC"]):
        print("Skipping", FNAME)
        continue

    COPYDIR = DIR + "/"
    os.makedirs(COPYDIR, exist_ok=True)
    try:
        paths = np.loadtxt(FNAME, dtype=str)
        paths = np.atleast_1d(paths)
        for path in paths:
            if not os.path.exists(path):
                print("DOES NOT EXIST!!", path)
                continue
            else:
                copyfile = (
                    COPYDIR
                    + f"/{BASENAME.split('popsummary_filepath_')[1].replace('txt', 'h5')}"
                )
                print("COPYING", path, copyfile)
                shutil.copy(path, copyfile)
    except Exception as e:
        print("FAILED!!", FNAME, e)
