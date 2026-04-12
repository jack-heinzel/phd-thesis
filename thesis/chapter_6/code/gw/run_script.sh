#!/bin/bash
# 43 2500 5
# ~/.conda/envs/gwjax311/bin/python3.11 inference.py --NPE 4000 --NVT 1000 --gpu 7 --vt_seed 0 --pe_seed 0 --Nevents 1000 --nlive 300

~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 1000 --nlive 300 --gpu 0 --pin_extra_params --pe_seed 0 --vt_seed 0 --uniform_prior
~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 1000 --nlive 300 --gpu 0 --pin_extra_params --pe_seed 0 --vt_seed 0 --uniform_prior --MinimumNeff 10
~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 1000 --nlive 300 --gpu 0 --pin_extra_params --pe_seed 0 --vt_seed 0 --uniform_prior --MinimumNeff 10 --downselect_to_new_neff 30

# ~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 16000 --NVT 4000 --nlive 500 --gpu 7 --pin_extra_params --pe_seed 0 --vt_seed 0

# ~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 100 --nlive 300 --gpu 0 --pin_extra_params --pe_seed 0 --vt_seed 0 --Nevents 200 --uniform_prior
# ~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 100 --nlive 300 --gpu 0 --pin_extra_params --pe_seed 0 --vt_seed 0 --Nevents 200 --uniform_prior --UncertaintyCut
