#!/bin/bash
# 43 2500 5

# ~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 1000 --nlive 300 --gpu 2 --pin_extra_params --pe_seed 1 --vt_seed 1 --uniform_prior
# ~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 1000 --nlive 300 --gpu 2 --pin_extra_params --pe_seed 1 --vt_seed 1 --uniform_prior --MinimumNeff 10
# ~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 1000 --nlive 300 --gpu 2 --pin_extra_params --pe_seed 1 --vt_seed 1 --uniform_prior --MinimumNeff 10 --downselect_to_new_neff 30
~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 1000 --nlive 300 --gpu 2 --pin_extra_params --pe_seed 1 --vt_seed 1 --uniform_prior --MinimumNeff 30 

# ~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 100 --nlive 300 --gpu 7 --pin_extra_params --pe_seed 2 --vt_seed 2 --Nevents 200 --uniform_prior
# ~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 100 --nlive 300 --gpu 7 --pin_extra_params --pe_seed 2 --vt_seed 2 --Nevents 200 --uniform_prior --UncertaintyCut

# ~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 100 --nlive 300 --gpu 7 --pin_extra_params --pe_seed 3 --vt_seed 3 --Nevents 200 --uniform_prior
# ~/.conda/envs/gwjax311/bin/python3.11 semi_parametric.py --NPE 4000 --NVT 100 --nlive 300 --gpu 7 --pin_extra_params --pe_seed 3 --vt_seed 3 --Nevents 200 --uniform_prior --UncertaintyCut


# ~/.conda/envs/gwjax311/bin/python3.11 new_flexible_model.py --NPE 16000 --NVT 4000 --gpu $1 --vt_seed 0 --pe_seed 0 --pin_extra_params --UncertaintyCut 
# ~/.conda/envs/gwjax311/bin/python3.11 new_flexible_model.py --NPE 16000 --NVT 4000 --gpu $1 --vt_seed 0 --pe_seed 0 --pin_extra_params 