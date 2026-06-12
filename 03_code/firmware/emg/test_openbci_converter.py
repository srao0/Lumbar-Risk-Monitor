import shutil
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.session_converter import (  # noqa: E402
    _resample_emg_dataframe,
    convert_session,
)


class OpenBCIConverterTests(unittest.TestCase):
    def setUp(self):
        self.work_dir = REPO_ROOT / ".test_work" / "test_openbci_converter"
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)
        self.work_dir.mkdir(parents=True)

    def tearDown(self):
        if self.work_dir.exists():
            shutil.rmtree(self.work_dir)

    def _write_openbci_csv(self, path: Path, fs: float, rows: int) -> None:
        t0 = 1000.0
        data = {
            "emg_ch1": [10.0 + i for i in range(rows)],
            "emg_ch2": [20.0 + i for i in range(rows)],
            "emg_ch3": [30.0 + i for i in range(rows)],
            "emg_ch4": [40.0 + i for i in range(rows)],
            "timestamp_unix": [t0 + (i / fs) for i in range(rows)],
        }
        pd.DataFrame(data).to_csv(path, index=False)

    def test_legacy_ganglion_alias_matches_generic_ganglion(self):
        raw_csv = self.work_dir / "ganglion_raw.csv"
        self._write_openbci_csv(raw_csv, fs=200.0, rows=400)

        legacy_out = self.work_dir / "legacy"
        generic_out = self.work_dir / "generic"
        convert_session(out_dir=legacy_out, ganglion_csv=raw_csv, phase="dev")
        convert_session(out_dir=generic_out, emg_csv=raw_csv, emg_board="ganglion", phase="dev")

        self.assertEqual(
            (legacy_out / "emg_data.csv").read_bytes(),
            (generic_out / "emg_data.csv").read_bytes(),
        )

    def test_cyton_250_hz_resamples_to_200_hz(self):
        raw_csv = self.work_dir / "cyton_raw.csv"
        self._write_openbci_csv(raw_csv, fs=250.0, rows=750)

        out_dir = self.work_dir / "cyton"
        convert_session(out_dir=out_dir, emg_csv=raw_csv, emg_board="cyton", phase="dev")
        emg_df = pd.read_csv(out_dir / "emg_data.csv")

        self.assertEqual(len(emg_df), 600)
        self.assertAlmostEqual(float(emg_df["timestamp_ms"].diff().dropna().median()), 5.0)

    def test_resampler_refuses_labelled_multi_segment_data(self):
        emg_df = pd.DataFrame(
            {
                "timestamp_ms": [0.0, 4.0, 8.0, 12.0],
                "label": ["A", "A", "B", "B"],
                "rep": [1, 1, 2, 2],
                "risk_class": [0, 0, 1, 1],
                "emg_LES_mv": [0.1, 0.2, 0.3, 0.4],
                "emg_RES_mv": [0.1, 0.2, 0.3, 0.4],
                "emg_LOBL_mv": [0.1, 0.2, 0.3, 0.4],
                "emg_ROBL_mv": [0.1, 0.2, 0.3, 0.4],
            }
        )

        with self.assertRaisesRegex(ValueError, "multiple 'label' values"):
            _resample_emg_dataframe(emg_df, target_fs=200.0)


if __name__ == "__main__":
    unittest.main()
