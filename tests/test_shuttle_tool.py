import csv
import tempfile
import unittest
from pathlib import Path
from src.shuttle_tool import evaluate, normalize_prediction_csv, save_rows


class ShuttleToolTests(unittest.TestCase):
    def test_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            save_rows(base/'ref.csv', [
                {'frame_index': 0, 'timestamp_seconds': 0, 'visible': 1, 'x': 10, 'y': 20, 'source': 'manual'},
                {'frame_index': 1, 'timestamp_seconds': .03, 'visible': 0, 'source': 'manual'},
            ])
            save_rows(base/'pred.csv', [
                {'frame_index': 0, 'timestamp_seconds': 0, 'visible': 1, 'x': 12, 'y': 21, 'source': 'model'},
                {'frame_index': 1, 'timestamp_seconds': .03, 'visible': 0, 'source': 'model'},
            ])
            report = evaluate(base/'ref.csv', base/'pred.csv')
            self.assertEqual((report['true_positive'], report['true_negative']), (1, 1))

    def test_import_tracknet(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base/'source.csv').write_text('Frame,Visibility,X,Y\n0,1,30,40\n1,0,0,0\n', encoding='utf-8')
            count = normalize_prediction_csv(base/'source.csv', base/'result.csv', 30, 476, 848)
            self.assertEqual(count, 2)
            with (base/'result.csv').open(newline='', encoding='utf-8') as fp:
                self.assertEqual(list(csv.DictReader(fp))[1]['visible'], '0')

    def test_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base/'source.csv').write_text('Frame,Visibility,X,Y\n0,1,999,40\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                normalize_prediction_csv(base/'source.csv', base/'result.csv', 30, 476, 848)


if __name__ == '__main__':
    unittest.main()
