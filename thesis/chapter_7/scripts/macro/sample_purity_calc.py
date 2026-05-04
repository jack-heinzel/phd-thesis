#!/usr/bin/env python3

# Determines the number of noise events in the O4a R&P paper sample

# -*- coding: utf-8 -*-
#
#       Copyright 2024
#       Adrian Helmling-Cornell 
#       Shio Sakon <shio.sakon@ligo.org>
#
#       This program is free software; you can redistribute it and/or modify
#       it under the terms of the GNU General Public License as published by
#       the Free Software Foundation; either version 2 of the License, or
#       (at your option) any later version.
#
#       This program is distributed in the hope that it will be useful,
#       but WITHOUT ANY WARRANTY; without even the implied warranty of
#       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#       GNU General Public License for more details.
#
#       You should have received a copy of the GNU General Public License
#       along with this program; if not, write to the Free Software
#       Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#       MA 02110-1301, USA.

import h5py
import json
import numpy as np

def far_per_yr_to_far_per_sec(far_thresh):
    return far_thresh / (86400 * 365.25)

def make_parser():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-hdf-filename",
        help="path to the input events hdf file",
        type=str,
        default="events.hdf",
    )

    parser.add_argument(
        "--remove-er15",
        help="remove er15 duration from O4a observing time",
        action='store_true'
    )

    parser.add_argument(
        "--output-filename",
        help="provide output json filename",
        type=str,
        default="samplePurityEstimate.json",
    )

    return parser

parser = make_parser()
args = parser.parse_args()

def num_events_per_far(far_thresh):  # Number of events in each epoch with a FAR less than the specified FAR threshold (in Hz)    
    num_gwtc_1_events = 0
    num_gwtc_2_events = 0
    num_gwtc_3_events = 0
    num_gwtc_4_events = 0
    with h5py.File(args.input_hdf_filename, 'r') as f:
        for event in f['O1']:
            if float(f['O1'][event]['FAR'][()]) <= far_thresh:
                num_gwtc_1_events += 1
        for event in f['O2']:
            if float(f['O2'][event]['FAR'][()]) <= far_thresh:
                num_gwtc_1_events += 1
        for event in f['O3a']:
            if float(f['O3a'][event]['FAR'][()]) <= far_thresh:
                num_gwtc_2_events += 1
        for event in f['O3b']:
            if float(f['O3b'][event]['FAR'][()]) <= far_thresh:
                num_gwtc_3_events += 1
        for event in f['O4a']:
            if float(f['O4a'][event]['FAR'][()]) <= far_thresh:
                num_gwtc_4_events += 1
    return num_gwtc_1_events, num_gwtc_2_events, num_gwtc_3_events, num_gwtc_4_events 

def num_false_per_catalog(num_analyses, far_thresh, observing_time):
    return num_analyses * far_thresh * observing_time

NUM_PIPELINES_IN_O1O2 = 2  # PyCBC, GstLAL, cWB operational (but cWB doesn't count for this)
NUM_PIPELINES_IN_O3 = 4  # MBTA, SPIIR operational too but catalog results use PyCBC, GstLAL, cWB and MBTA
NUM_PIPELINES_IN_O4A = 4 # MBTA, SPIIR, PyCBC, GstLAL, cWB but catalog results use PyCBC, GstLAL, cWB and MBTA

OBSERVING_TIME_O1O2 = .46 * 365.25 * 86400 # seconds, from GWTC-1 Section VII (and https://git.ligo.org/publications/O2/cbc-catalog/-/blob/master/macros/General_macros.tex)
OBSERVING_TIME_O3A = 15843600  # s, from P2000217
OBSERVING_TIME_O3B = 12905976 # s, from T2100323
if args.remove_er15:
    OBSERVING_TIME_O4A = 17019212 # s, (1389456018−1368975618)  FIXME: does not include ER15 
else:
    OBSERVING_TIME_O4A = 20837891 # s, (1368552711-1368195220) + (1389456018−1368975618), see https://dcc.ligo.org/M2400111, FIXME
PAPER_FAR_THRESHOLD = far_per_yr_to_far_per_sec(1)  # FAR of 1 per 1 year, converted to FAR per 1 second

num_false_gwtc1 = num_false_per_catalog(NUM_PIPELINES_IN_O1O2, PAPER_FAR_THRESHOLD, OBSERVING_TIME_O1O2)
num_false_gwtc2 = num_false_per_catalog(NUM_PIPELINES_IN_O3, PAPER_FAR_THRESHOLD, OBSERVING_TIME_O3A)
num_false_gwtc3 = num_false_per_catalog(NUM_PIPELINES_IN_O3, PAPER_FAR_THRESHOLD, OBSERVING_TIME_O3B)
num_false_gwtc4 = num_false_per_catalog(NUM_PIPELINES_IN_O4A, PAPER_FAR_THRESHOLD, OBSERVING_TIME_O4A)
total_num_false = num_false_gwtc1 + num_false_gwtc2 + num_false_gwtc3 + num_false_gwtc4

num_false_gwtc1_simple = num_false_per_catalog(1, PAPER_FAR_THRESHOLD, OBSERVING_TIME_O1O2)
num_false_gwtc2_simple = num_false_per_catalog(1, PAPER_FAR_THRESHOLD, OBSERVING_TIME_O3A)
num_false_gwtc3_simple = num_false_per_catalog(1, PAPER_FAR_THRESHOLD, OBSERVING_TIME_O3B)
num_false_gwtc4_simple = num_false_per_catalog(1, PAPER_FAR_THRESHOLD, OBSERVING_TIME_O4A)
total_num_false_simple = num_false_gwtc1_simple + num_false_gwtc2_simple + num_false_gwtc3_simple + num_false_gwtc4_simple

num_events_gwtc1, num_events_gwtc2, num_events_gwtc3, num_events_gwtc4 = num_events_per_far(PAPER_FAR_THRESHOLD)
total_num_events = num_events_gwtc1 + num_events_gwtc2 + num_events_gwtc3 + num_events_gwtc4
num_events_gwtc1_far4, num_events_gwtc2_far4, num_events_gwtc3_far4, num_events_gwtc4_far4 = num_events_per_far(far_per_yr_to_far_per_sec(1. / 4))
total_num_events_with_far4 = num_events_gwtc1_far4 + num_events_gwtc2_far4 + num_events_gwtc3_far4 + num_events_gwtc4_far4
total_analysis_time = (OBSERVING_TIME_O1O2 + OBSERVING_TIME_O3A + OBSERVING_TIME_O3B + OBSERVING_TIME_O4A) / (86400 * 365.25)

print('The number of false events expected in the dataset is {:.1f}. This represents {:.2f}% of the events considered in this paper. Without considering a trials factor for number of pipelines, the number of false events is instead {:.1f}. This represents {:.2f}% of the events considered in this paper.'.format(total_num_false, 100. * total_num_false / total_num_events, total_num_false_simple, 100. * total_num_false_simple / total_num_events))

data = {'totalNumEvents' : total_num_events, 
	'numEventsFAROnePerFourYear' : total_num_events_with_far4,
	'expectedNumFalseTrialsFactor' : {'full' : total_num_false, 'rounded' : np.round(total_num_false, 1)},
	'expectedNumFalseSimple' :  {'full' : total_num_false_simple, 'rounded' : np.round(total_num_false_simple, 1)},
	'percentFalseTrialsFactor' : {'full' : 100. * total_num_false / total_num_events, 'rounded' : np.round(100. * total_num_false / total_num_events, 2)},
	'percentFalseSimple' : {'full' : 100. * total_num_false_simple / total_num_events, 'rounded' : np.round(100. * total_num_false_simple / total_num_events, 2)},
	'totalAnalysisTimeYears' : {'full' : total_analysis_time, 'rounded' : np.round(total_analysis_time, 2)},
	'totalNumEventsO4aOnly' : num_events_gwtc4,
	'totalNumEventsGWTC3' : (num_events_gwtc1 + num_events_gwtc2 + num_events_gwtc3)
}

with open(args.output_filename, 'w') as f:
    json.dump(data, f)
