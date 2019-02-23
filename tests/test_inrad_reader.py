#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from inrad_reader.inrad_to_cf import read_multi_radar
import pyart

__author__ = "Joseph C. Hardin"
__copyright__ = "Joseph C. Hardin"
__license__ = "mit"

test_file_path = '/Volumes/hard_lacie_hfs/data/indian_radar_data/'

def test_inrad_reader_lists_files():
    radar = read_multi_radar(test_file_path + 'T_HAHA00_C_DEMS_20180701080230*')
    assert(type(radar) == pyart.core.radar.Radar)
