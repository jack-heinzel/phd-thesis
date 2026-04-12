#!/bin/bash

# 10 12 13 14 15 17 19 20 22 25 27 30 33 36 39 43 47 52 57 63 69 75 83 91 100 109 120 131 144 158 173 190 208 229 251 275 301 331 363 398 436 478 524 575 630 691 758 831 912 1000
for npe in 173 190 208
do
    /home/jack.heinzel/.conda/envs/gwjax311/bin/python3.11 /home/jack.heinzel/public_html/bias_2024/hierarchical_posteriors/one_dimensional/hierarchical_inference.py --gpus $1 --npe ${npe} --nrandom 100000 --noise_sd 1 --cov_batch_size 10
done