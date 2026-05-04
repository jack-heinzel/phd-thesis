#!/usr/bin/env python3

# Determines the number of events in the O4a R&P paper sample for different thresholds

# -*- coding: utf-8 -*-
#
#       Copyright 2025
#       Ben Farr
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
import xarray as xr


far_thresholds = [1.0, 0.25]
sources = ["BNS", "NSBH", "BBH"]
obs_runs = ["O1", "O2", "O3a", "O3b", "O4a"]
this_obs_run = "O4a"


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
        "--output-filename",
        help="provide output json filename",
        type=str,
        default="catalogCounts.json",
    )

    return parser


parser = make_parser()
args = parser.parse_args()


def counts_per_far(far_thresholds=far_thresholds, obs_runs=obs_runs, sources=sources):
    counts = np.zeros((len(far_thresholds), len(obs_runs), len(sources)))
    with h5py.File(args.input_hdf_filename, 'a') as f:
        for t, far_threshold in enumerate(far_thresholds):
            for o, obs_run in enumerate(obs_runs):
                for event in f[obs_run]:
                    if float(f[obs_run][event]['FAR'][()]) <= far_per_yr_to_far_per_sec(far_threshold):
                        #if 'cwb' in str(f[obs_run][event]['pipeline'][()]):
                            # print("skipping ", f[obs_run][event]['GWname'][()].decode(), "...")
                            # print("   pipeline: ", f[obs_run][event]['pipeline'][()].decode())
                            #continue
                        cat_probs = [f[obs_run][event]['p_{}'.format(source)][()] for source in sources]
                        if 'GW230529' in str(f[obs_run][event]['GWname'][()]):  # MBTA reports as BNS, but likely NSBH
                            print(far_threshold, str(f[obs_run][event]['GWname'][()]))
                            counts[t][o][sources.index('NSBH')] += 1
                            if far_threshold == 1.0:
                                try:
                                    f.create_dataset('%s/%s/category' % (obs_run, event), data='NSBH')
                                except ValueError:
                                    pass
                        elif 'GW190917' in str(f[obs_run][event]['GWname'][()]): # pastro suggests it's a BBH but it's a NSBH
                            print(far_threshold, str(f[obs_run][event]['GWname'][()]))
                            counts[t][o][sources.index('NSBH')] += 1
                            if far_threshold == 1.0:
                                try:
                                    f.create_dataset('%s/%s/category' % (obs_run, event), data='NSBH')
                                except ValueError:
                                    pass
                        elif 'cwb' in str(f[obs_run][event]['pipeline'][()]):
                            counts[t][o][sources.index('BBH')] += 1
                            if far_threshold == 1.0:
                                try:
                                    f.create_dataset('%s/%s/category' % (obs_run, event), data='BBH')
                                except ValueError:
                                    pass
                        elif 'GW190814' in str(f[obs_run][event]['GWname'][()]):  # treat as outlier; don't count
                            # counts[t][o][sources.index('NSBH')] += 1
                            if far_threshold == 1.0:
                                try:
                                    f.create_dataset('%s/%s/category' % (obs_run, event), data='???')
                                except ValueError:
                                    f[obs_run][event]['category'][()] = '???'

                        else:
                            counts[t][o][np.argmax(cat_probs)] += 1
                            if np.argmax(cat_probs) == 1:
                                print(far_threshold, str(f[obs_run][event]['GWname'][()]))
                            if far_threshold == 1.0:
                                if np.argmax(cat_probs) == 2:
                                    try:
                                        f.create_dataset('%s/%s/category' % (obs_run, event), data='BBH')
                                    except ValueError:
                                        pass
                                elif np.argmax(cat_probs) == 1:
                                    try:
                                        f.create_dataset('%s/%s/category' % (obs_run, event), data='NSBH')
                                    except ValueError:
                                        pass
                                elif np.argmax(cat_probs) == 0:
                                    try:
                                        f.create_dataset('%s/%s/category' % (obs_run, event), data='BNS')
                                    except ValueError:
                                        pass

    xcounts = xr.DataArray(
        counts,
        dims=("far_threshold", "obs_run", "source"),
        coords={"far_threshold": far_thresholds, "obs_run": obs_runs, "source": sources},
    )
    return xcounts


counts = counts_per_far()
data = {}
for far, far_label in zip([1.0, 0.25], ['FAROnePerYear', 'FAROnePerFourYears']):
    data[far_label] = {
        'O4aBNS': int(counts.sel(far_threshold=far, obs_run='O4a', source='BNS').values),
        'O4aNSBH': int(counts.sel(far_threshold=far, obs_run='O4a', source='NSBH').values),
        'O4aBBH': int(counts.sel(far_threshold=far, obs_run='O4a', source='BBH').values),
        'O4aTotal': int(counts.sel(far_threshold=far, obs_run='O4a').sum().values),
        'O3bBNS': int(counts.sel(far_threshold=far, obs_run='O3b', source='BNS').values),
        'O3bNSBH': int(counts.sel(far_threshold=far, obs_run='O3b', source='NSBH').values),
        'O3bBBH': int(counts.sel(far_threshold=far, obs_run='O3b', source='BBH').values),
        'O3bTotal': int(counts.sel(far_threshold=far, obs_run='O3b').sum().values),
        'O3aBNS': int(counts.sel(far_threshold=far, obs_run='O3a', source='BNS').values),
        'O3aNSBH': int(counts.sel(far_threshold=far, obs_run='O3a', source='NSBH').values),
        'O3aBBH': int(counts.sel(far_threshold=far, obs_run='O3a', source='BBH').values),
        'O3aTotal': int(counts.sel(far_threshold=far, obs_run='O3a').sum().values),
        'O2BNS': int(counts.sel(far_threshold=far, obs_run='O2', source='BNS').values),
        'O2NSBH': int(counts.sel(far_threshold=far, obs_run='O2', source='NSBH').values),
        'O2BBH': int(counts.sel(far_threshold=far, obs_run='O2', source='BBH').values),
        'O2Total': int(counts.sel(far_threshold=far, obs_run='O2').sum().values),
        'O1BNS': int(counts.sel(far_threshold=far, obs_run='O1', source='BNS').values),
        'O1NSBH': int(counts.sel(far_threshold=far, obs_run='O1', source='NSBH').values),
        'O1BBH': int(counts.sel(far_threshold=far, obs_run='O1', source='BBH').values),
        'O1Total': int(counts.sel(far_threshold=far, obs_run='O1').sum().values),
        'totalBNS': int(counts.sel(far_threshold=far, source='BNS').sum().values),
        'totalNSBH': int(counts.sel(far_threshold=far, source='NSBH').sum().values),
        'totalBBH': int(counts.sel(far_threshold=far, source='BBH').sum().values),
        'total': int(counts.sel(far_threshold=far).sum().values),
    }
data['combined'] = {
    'total': data['FAROnePerFourYears']['totalBNS'] + data['FAROnePerFourYears']['totalNSBH'] + data['FAROnePerYear']['totalBBH']
}

with open(args.output_filename, 'w') as f:
    json.dump(data, f)
