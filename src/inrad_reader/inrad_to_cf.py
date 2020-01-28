#!/usr/bin/env python
# -*- coding: utf-8 -*-
import xarray as xr
import pandas as pd
import numpy as np
import glob
import pyart
import tempfile
from datetime import datetime
from datetime import timezone

from operator import itemgetter
from itertools import groupby
import argparse
import sys
import logging
import json
import sklearn.cluster
from default_config import default_config as default_config

from inrad_reader import __version__

__author__ = "Joseph C. Hardin"
__copyright__ = "Joseph C. Hardin"
__license__ = "mit"

_logger = logging.getLogger(__name__)

SCAN_TYPE_MAPPING = {
    0: 'other',
    1: 'sector',
    2: 'rhi',
    4: 'ppi',
    7: 'rhi'
}

MOMENT_NAME_MAPPING = {
    'T': 'total_power',
    'Z': 'reflectivity',
    'V': 'mean_doppler_velocity',
    'W': 'spectrum_width',
    'ZDR': 'differential_reflectivity',
    'KDP': "specific_differential_phase",
    "PHIDP": "differential_phase",
    "SQI": "normalized_coherent_power",
    "RHOHV": "copol_correlation_coeff",
    "HCLASS": "hydrometeor_classification"
}


def get_sweep_num_from_filename(filename):
    parts = filename.split('sweep')
    return int(parts[1].split('.nc')[0])


def get_sorted_list(filename_list_glob):
    pass
    filename_list = glob.glob(filename_list_glob)
    filename_list.sort(key=get_sweep_num_from_filename)
    return filename_list


def parse_args(args):
    """Parse command line parameters

    Args:
      args ([str]): command line parameters as list of strings

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(
        description="Indian weather radar reader, merger, and gridder.")
    parser.add_argument(
        '--version',
        action='version',
        version='inrad_reader {ver}'.format(ver=__version__))
    parser.add_argument(
        '-fg',
        '--file_glob',
        dest="file_glob",
        help="file glob used to collect files",
        type=str,
        metavar="file_glob",
        required=True)
    parser.add_argument(
        '-or',
        '--output_radial',
        dest="output_filename_radial",
        help="Output Filename for CF/Radial file",
        type=str,
        metavar="output_filename_radial",
        default=False
        )
    parser.add_argument(
        '-og',
        '--output_gridded',
        dest="output_filename_gridded",
        help="Output filename for gridded file",
        type=str,
        metavar="output_filename_gridded",
        default=False
    )
    parser.add_argument(
        '-v',
        '--verbose',
        dest="loglevel",
        help="set loglevel to INFO",
        action='store_const',
        const=logging.INFO)
    parser.add_argument(
        '-vv',
        '--very-verbose',
        dest="loglevel",
        help="set loglevel to DEBUG",
        action='store_const',
        const=logging.DEBUG)
    parser.add_argument(
        '-c',
        '--config',
        dest="config_file",
        help="Configuration file",
        metavar="config_file",
        type=str,
        default=False
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(level=loglevel, stream=sys.stdout,
                        format=logformat, datefmt="%Y-%m-%d %H:%M:%S")


def main(args):
    """Main entry point allowing external calls

    Args:
      args ([str]): command line parameter list
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    filename_glob = args.file_glob
    config = False

    if args.config_file:
        print(args.config_file)
        config = json.load(open(args.config_file))
    else:
        config = default_config

    _logger.debug("Config used:")
    _logger.debug(config)

    radar = read_multi_radar(args.file_glob)
    
    # extract the elevation for each sweep (modified by Jingyi Chen)
    nsweep = np.max(radar.sweep_number['data'])+1 # number of sweeps
    elevation_data = radar.elevation['data']
    kmeans_model = sklearn.cluster.KMeans(n_clusters=nsweep, random_state=0).fit(elevation_data.reshape(-1, 1))
    ele_clusters = kmeans_model.fit_predict(elevation_data.reshape(-1, 1))
    elevation_sweep = np.zeros(nsweep) # elevation for each sweep
    for ii in range(nsweep):
        unique, counts = np.unique(elevation_data[ele_clusters==ii], return_counts=True)
        elevation_sweep[ii] = unique[np.argmax(counts)]
    elevation_sweep = np.sort(elevation_sweep)   

    # write to netcdf files
    if args.output_filename_radial:
        _logger.debug("Writing CF/Radial file")
        pyart.io.write_cfradial(args.output_filename_radial, radar)
    

    if args.output_filename_gridded:
        gatefilter = pyart.filters.GateFilter(radar)
        gatefilter.exclude_invalid('reflectivity')

        _logger.debug("Gridding radar file")
        config['grid_shape'] = tuple(config['grid_shape']) #Fix a weird pyart bug.

        field_list = []
        if 'fields' not in config:
            for moment in MOMENT_NAME_MAPPING.values():
                if moment in radar.fields:
                    field_list.append(moment)
                else:
                    continue
            config['fields'] = field_list

        print(config['fields'])

        grid = pyart.map.grid_from_radars(radar, **config)

        _logger.debug("Writting Gridded file")
#         pyart.io.write_grid(args.output_filename_gridded, grid, 
#                                write_point_lon_lat_alt=config['write_point_lon_lat'])
        temp_file = tempfile.NamedTemporaryFile()
        pyart.io.write_grid(temp_file.name, grid, write_point_lon_lat_alt=config['write_point_lon_lat'])
        gridradObj = xr.open_dataset(temp_file.name, decode_times=False)   
        gridradObj.attrs['nsweep'] = nsweep
        gridradObj.expand_dims({'nsweep':int(nsweep)})
        gridradObj['elevation_sweep'] = (('nsweep'), elevation_sweep)
        gridradObj.expand_dims({'unique_elevation':unique.shape[0]})
        gridradObj['elevations'] = (('unique_elevation',unique))
        gridradObj['counts_elevations'] = (('unique_elevation',counts))
        comp = dict(zlib=True) # Set encoding/compression for all variables
        encoding = {var: comp for var in gridradObj.data_vars}
        gridradObj.to_netcdf(path=args.output_filename_gridded, mode='w', format='NETCDF4_CLASSIC', unlimited_dims='time', \
                        encoding=encoding, )
        


def read_multi_radar(filename_glob):
    """ Take a list of filenames and build up a pyart radar object

    Parameters:
    -----------
    filename_glob: str
        Regex to grab filenames to merge into one volume.
    """

    filename_list = get_sorted_list(filename_glob)
    _logger.debug(f"Found {len(filename_list)} files to process.")
    dset = xr.open_mfdataset(filename_list, concat_dim='radial', data_vars='different', decode_times=False)

    _range = {'data': np.arange(0, dset.dims['bin']) * dset['gateSize'].values + dset['firstGateRange'].values}
    fixed_angle = np.array(list(map(itemgetter(0), groupby(dset['elevationAngle'].values))))
    num_rays_per_sweep = [np.count_nonzero(np.isclose(dset['elevationAngle'].values, val)) for val in fixed_angle]
    sweep_start_ray_index = {'data': np.hstack((0, np.cumsum(num_rays_per_sweep)[0:-1]))}
    sweep_end_ray_index = {'data': (np.cumsum(num_rays_per_sweep) - 1)}
    time = {'data': np.array(dset['radialTime']),
            'units': 'seconds since 1970-1-1 00:00:00.00'}
    fields = {}

    for moment_name in MOMENT_NAME_MAPPING:
        if moment_name in dset:
            fields[MOMENT_NAME_MAPPING[moment_name]] = {
                'data': dset[moment_name].values,
                'units': dset[moment_name].units,
                'long_name': dset[moment_name].long_name
            }
    sweep_mode = []
    for _ in np.arange(dset.dims['sweep']):
        sweep_mode.append('manual_' + SCAN_TYPE_MAPPING[int(dset['scanType'].values)])
    sweep_mode = np.array(sweep_mode)

    radar = pyart.core.Radar(time=time, _range=_range, azimuth={'data': dset['radialAzim'].values},
                             elevation={'data': dset['radialElev'].values},
                             fixed_angle={'data': fixed_angle}, sweep_start_ray_index=sweep_start_ray_index,
                             sweep_end_ray_index=sweep_end_ray_index,
                             longitude={'data': np.array([float(dset['siteLon'].values), ])},
                             latitude={'data': np.array([float(dset['siteLat'].values), ])},
                             altitude={'data': np.array([float(dset['siteAlt'].values), ])},
                             scan_type=SCAN_TYPE_MAPPING[int(dset['scanType'].values)],
                             sweep_number={'data': np.arange(0, dset.dims['sweep'])},
                             fields=fields, metadata={}, sweep_mode={'data': sweep_mode}
                             )

    return radar


def run():
    """Entry point for console_scripts
    """

    main(sys.argv[1:])


if __name__ == "__main__":
    run()
