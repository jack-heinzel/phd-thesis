#!/usr/bin/env python3

# -*- coding: utf-8 -*-
#
#       Copyright 2020
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

import numpy as np
from gwpy.time import tconvert
from gwpy.table import EventTable
import glob
import h5py

print('Finished imports!')

def make_parser():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gwtc1-path",
        help="path to gwtc-1 folder",
        type=str,
    )

    parser.add_argument(
        "--gwtc2-path",
        help="path to gwtc-2 folder",
        type=str,
    )

    parser.add_argument(
        "--gwtc3-path",
        help="path to gwtc-3 folder",
        type=str,
    )

    parser.add_argument(
        "--gwtc4-path",
        help="path to gwtc-4 folder",
        type=str,
    )

    parser.add_argument(
        "--far-threshold",
        help="max far to include, in 1/years",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--output-file-name",
        help="name of output hdf filename",
        type=str,
        default="events.hdf"
    )

    parser.add_argument(
        "--remove-er15-events",
        help="remove ER15 events from the event list",
        action='store_true'
    )

    return parser

parser = make_parser()
args = parser.parse_args() 

def far_per_yr_to_far_per_sec(far_thresh):
    return far_thresh / (86400 * 365.25)

def query_gwtcs(gwtc_path):
    data_path = gwtc_path
    search_file = glob.glob(data_path+"search_results/*SearchSummaryTable.hdf5")[0]
    search_summary = EventTable.read(search_file, path="search_summary")
    search_summary = search_summary.filter("dq_status != DQ_veto")
    far_thresh = far_per_yr_to_far_per_sec(args.far_threshold) 
    high_sig_events = search_summary.filter(f"far < {far_thresh}")
    return high_sig_events

def dump_to_hdf5(filename, gwtc, high_sig_events):
    with h5py.File(filename, 'a') as f:
        for event in high_sig_events:
            if gwtc=="GWTC1":
                if event['gps_time'] <= 1137254417:
                    observing_run = "O1"
                elif event['gps_time'] > 1137254417:
                    observing_run = "O2"
            elif gwtc=="GWTC2":
                observing_run = "O3a"
            elif gwtc=="GWTC3":
                observing_run = "O3b"
            elif gwtc=="GWTC4":
                observing_run = "O4a"
            base_path = f"{observing_run}/{event['gw_name']}"
            if base_path in f:
                print(f"Skipping {base_path} (already exists)")
                continue
            f.create_dataset('%s/%s/GWname' % (observing_run, event['gw_name']), data=event['gw_name'])
            f.create_dataset('%s/%s/FAR' % (observing_run, event['gw_name']), data=event['far'])
            f.create_dataset('%s/%s/SNR' % (observing_run, event['gw_name']), data=event['snr'])
            f.create_dataset('%s/%s/gps_time' % (observing_run, event['gw_name']), data=event['gps_time'])
            f.create_dataset('%s/%s/pipeline' % (observing_run, event['gw_name']), data=event['pipeline'])
            f.create_dataset('%s/%s/p_BBH' % (observing_run, event['gw_name']), data=event['p_BBH'])
            f.create_dataset('%s/%s/p_BNS' % (observing_run, event['gw_name']), data=event['p_BNS'])
            f.create_dataset('%s/%s/p_NSBH' % (observing_run, event['gw_name']), data=event['p_NSBH'])

# GWTC1
print('Parsing GWTC1')
high_sig_events = query_gwtcs(args.gwtc1_path)
dump_to_hdf5(args.output_file_name, 'GWTC1', high_sig_events) 

# GWTC2
print('Parsing GWTC2')
high_sig_events = query_gwtcs(args.gwtc2_path)
dump_to_hdf5(args.output_file_name, 'GWTC2', high_sig_events) 

# GWTC3
print('Parsing GWTC3')
high_sig_events = query_gwtcs(args.gwtc3_path)
dump_to_hdf5(args.output_file_name, 'GWTC3', high_sig_events) 

# GWTC4 
print('Parsing GWTC4')
high_sig_events = query_gwtcs(args.gwtc4_path)
dump_to_hdf5(args.output_file_name, 'GWTC4', high_sig_events) 

# Remove human-vetted, CWB-only, and ER15  events 
with h5py.File(args.output_file_name, 'a') as f:
    del f['O3a/GW190531_023648']
    #del f['O3b/GW200219_201407']
    #del f['O3b/GW200214_224526']
    if args.remove_er15_events:
        del f['O4a/GW230518_125908']

print('Done!')
