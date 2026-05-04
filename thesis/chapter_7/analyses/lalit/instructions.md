popmodel: peakcut
otherinfo: excluding GW190814 event
filepath: /home/lalit.pathak/O4/o4a-ns-pop/O4a/production_runs/for_o4a_astro_paper/18nov_rerun/peakcut_m1m2_woGW190814/peakcut_m1m2.h5

popmodel: peakcut
otherinfo: including GW190814 event
filepath: /home/lalit.pathak/O4/o4a-ns-pop/O4a/production_runs/for_o4a_astro_paper/18nov_rerun/peakcut_m1m2_wGW190814/peakcut_m1m2.h5

popmodel: power
otherinfo: excluding GW190814 event
filepath: /home/lalit.pathak/O4/o4a-ns-pop/O4a/production_runs/for_o4a_astro_paper/18nov_rerun/power_m1m2_woGW190814/power_m1m2.h5

popmodel: power
otherinfo: including GW190814 event
filepath: /home/lalit.pathak/O4/o4a-ns-pop/O4a/production_runs/for_o4a_astro_paper/18nov_rerun/power_m1m2_wGW190814/power_m1m2.h5

##---instructions for running sodapop_macros.py file ---##
## We need to provide the path to the basedirectory and NS mass population model assuming you are in an environment which has sodapop installed
python3 sodapop_macros.py --basedirectory /home/lalit.pathak/O4/o4a-ns-pop/O4a/production_runs/for_o4a_astro_paper/18nov_rerun/ --popmodel peakcut
python3 sodapop_macros.py --basedirectory /home/lalit.pathak/O4/o4a-ns-pop/O4a/production_runs/for_o4a_astro_paper/18nov_rerun/ --popmodel power