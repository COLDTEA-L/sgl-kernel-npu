"""No-card dispatcher checks; these do not replace two-rank device tests."""

import ctypes
import unittest

import torch
import deep_ep.deep_ep_cpp as extension
from deep_ep import Buffer


class PreparedMultipathMetaTest(unittest.TestCase):
    def setUp(self):
        self.op = torch.ops.deep_ep.ccu_urma_prepared_multipath_alltoall_policy
        self.x = torch.empty((2, 1024), device="meta", dtype=torch.float32)

    def test_installed_abi_and_apis(self):
        library = ctypes.CDLL(extension.__file__)
        version = library.A5DeepEpExplicitMultipathAttrAbiVersion
        version.restype = ctypes.c_int
        self.assertGreaterEqual(version(), 11)
        self.assertTrue(hasattr(Buffer, "bind_ccu_urma_explicit_multipath_plan"))
        self.assertTrue(hasattr(Buffer, "ccu_urma_prepared_multipath_alltoall_policy_out"))

    def test_supported_path_counts_and_output(self):
        for relays in range(1, 13):
            y = self.op(self.x, 1, [2] + [1] * relays)
            self.assertEqual(y.device.type, "meta")
            self.assertEqual(y.shape, self.x.shape)
            self.assertEqual(y.dtype, self.x.dtype)

    def test_invalid_weights(self):
        for weights in ([], [1], [2, 0], [2, -1], [2, 2**32], [1] * 14):
            with self.subTest(weights=weights), self.assertRaises(RuntimeError):
                self.op(self.x, 1, weights)

    def test_invalid_tensor_and_handle(self):
        for x in (
            torch.empty((3, 1024), device="meta"),
            torch.empty((2, 0), device="meta"),
            torch.empty((2, 1024), device="meta", dtype=torch.float16),
            torch.empty((1024, 2), device="meta").t(),
        ):
            with self.subTest(shape=x.shape, dtype=x.dtype), self.assertRaises(RuntimeError):
                self.op(x, 1, [2, 1])
        with self.assertRaises(RuntimeError):
            self.op(self.x, 0, [2, 1])

    def test_fullgraph_meta_capture(self):
        def execute(x):
            return self.op(x, 1, [2, 1, 1])

        captured = torch.compile(execute, backend="eager", fullgraph=True)
        self.assertEqual(captured(self.x).shape, self.x.shape)


if __name__ == "__main__":
    print("loaded DeepEP:", extension.__file__)
    unittest.main()
