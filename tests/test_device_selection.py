import unittest
from unittest.mock import patch

from zipvoice.luxvoice import _resolve_device


class ResolveDeviceTest(unittest.TestCase):
    @patch("zipvoice.luxvoice.torch.backends.mps.is_available", return_value=False)
    @patch("zipvoice.luxvoice._xpu_is_available", return_value=True)
    @patch("zipvoice.luxvoice.torch.cuda.is_available", return_value=False)
    def test_cuda_falls_back_to_xpu(self, _cuda_available, _xpu_available, _mps_available):
        self.assertEqual(_resolve_device("cuda"), "xpu")
        self.assertEqual(_resolve_device("cuda:1"), "xpu")

    @patch("zipvoice.luxvoice.torch.backends.mps.is_available", return_value=True)
    @patch("zipvoice.luxvoice._xpu_is_available", return_value=False)
    @patch("zipvoice.luxvoice.torch.cuda.is_available", return_value=False)
    def test_cuda_falls_back_to_mps(self, _cuda_available, _xpu_available, _mps_available):
        self.assertEqual(_resolve_device("cuda"), "mps")

    @patch("zipvoice.luxvoice.torch.backends.mps.is_available", return_value=False)
    @patch("zipvoice.luxvoice._xpu_is_available", return_value=False)
    @patch("zipvoice.luxvoice.torch.cuda.is_available", return_value=False)
    def test_cuda_falls_back_to_cpu(self, _cuda_available, _xpu_available, _mps_available):
        self.assertEqual(_resolve_device("cuda"), "cpu")

    @patch("zipvoice.luxvoice._xpu_is_available", return_value=True)
    def test_indexed_xpu_is_preserved(self, _xpu_available):
        self.assertEqual(_resolve_device("xpu:0"), "xpu:0")

    @patch("zipvoice.luxvoice._xpu_is_available", return_value=False)
    def test_unavailable_xpu_is_rejected(self, _xpu_available):
        with self.assertRaisesRegex(RuntimeError, "XPU was requested but is unavailable"):
            _resolve_device("xpu")


if __name__ == "__main__":
    unittest.main()
