"""UMFPACK helper: 64-bit indices with numpy >= 2."""
import unittest

import numpy as np
import scipy.sparse as sp

from pymmr.finite_volume import _get_umf_family


class TestUmfFamily(unittest.TestCase):

    def test_int32_indices(self):
        # int32 indices must be converted to int64 (np.array(copy=False) raised with numpy >= 2)
        A = sp.csr_array(sp.random(20, 20, density=0.2, random_state=0) + sp.eye(20))
        A.indptr = A.indptr.astype(np.int32)
        A.indices = A.indices.astype(np.int32)
        family, A_new = _get_umf_family(A)
        self.assertEqual(family, 'dl')
        self.assertEqual(A_new.indptr.dtype, np.int64)
        self.assertEqual(A_new.indices.dtype, np.int64)
        self.assertEqual(A.indices.dtype, np.int32)    # A is not modified
        np.testing.assert_array_equal(A_new.toarray(), A.toarray())


if __name__ == '__main__':
    unittest.main()
