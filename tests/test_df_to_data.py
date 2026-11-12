#!/usr/bin/env python3
"""Tests of inversion.df_to_data (DataFrame to DataMMR / DataERT)."""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pymmr.inversion import df_to_data  # noqa: E402

N = 5
DATES = np.array(['2024-05', '2023-09', '2024-05', '2023-09', '2024-05'])


def frame(kind):
    i = np.arange(N, dtype=float)
    d = {}
    for e, k in (('c1', 1.0), ('c2', 2.0)):
        for c in 'xyz':
            d[e + '_' + c] = k * 100.0 + i
    if kind == 'mmr':
        for c in 'xyz':
            d['obs_' + c] = i + 0.5
        for n, c in enumerate('xyz'):
            d['B' + c] = 10.0 * i + n          # datum i, component n
            d['wt_' + c] = 100.0 * i + n
    else:
        for e in ('p1', 'p2'):
            for c in 'xyz':
                d[e + '_' + c] = i + 0.5
        d['V'] = 10.0 * i
        d['wt'] = 100.0 * i
    d['cs'] = 1.0 + i
    return pd.DataFrame(d)


class TestDfToData(unittest.TestCase):

    def test_mmr(self):
        d = df_to_data(frame('mmr'))
        self.assertIsNone(d.date)
        i = np.arange(N)
        np.testing.assert_array_equal(d.data, 10.0 * i[:, None] + np.arange(3))
        np.testing.assert_array_equal(d.wt, (100.0 * i[None, :] + np.arange(3)[:, None]).ravel())

    def test_mmr_dates(self):
        df = frame('mmr')
        df['date'] = DATES
        out = df_to_data(df)
        self.assertEqual([v.date for v in out], ['2023-09', '2024-05'])
        for v in out:
            i = np.flatnonzero(DATES == v.date)
            np.testing.assert_array_equal(v.xs[:, 0], 100.0 + i)
            np.testing.assert_array_equal(v.xo[:, 0], i + 0.5)
            np.testing.assert_array_equal(v.data, 10.0 * i[:, None] + np.arange(3))
            np.testing.assert_array_equal(v.cs, 1.0 + i)
            # all x, then all y, then all z, as without dates
            np.testing.assert_array_equal(v.wt, (100.0 * i[None, :] + np.arange(3)[:, None]).ravel())

    def test_ert_dates(self):
        df = frame('ert')
        df['date'] = DATES
        out = df_to_data(df)
        self.assertEqual([v.date for v in out], ['2023-09', '2024-05'])
        for v in out:
            i = np.flatnonzero(DATES == v.date)
            np.testing.assert_array_equal(v.c1c2[:, 3], 200.0 + i)
            np.testing.assert_array_equal(v.data, 10.0 * i)
            np.testing.assert_array_equal(v.wt, 100.0 * i)
            np.testing.assert_array_equal(v.cs, 1.0 + i)


if __name__ == '__main__':
    unittest.main()
