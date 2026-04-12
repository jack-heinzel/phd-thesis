#!/bin/bash

/home/jack.heinzel/.conda/envs/gwjax311/bin/python3.11 /home/jack.heinzel/public_html/bias_2024/mc-bias/code/gw/single-event-pe/runPE.py --index $1 --nlive 1024 --relativeBinning --seed $2
