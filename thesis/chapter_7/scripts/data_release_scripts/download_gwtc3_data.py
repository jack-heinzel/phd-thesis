import os
import shutil

if os.path.exists('gwtc3_data'):
    print("Directory already exists, overwriting with newly downloaded files")
    shutil.rmtree("gwtc3_data")

for fname in ["analyses_PowerLawPeak.tar.gz", "analyses_GaussianSpin.tar.gz"]:
    os.system(f"zenodo_get 11254021 -g {fname} -o gwtc3_data/")
    os.system(f"tar -xvzf gwtc3_data/{fname} -C gwtc3_data/")
