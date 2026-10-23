"""Regression for the first real CIFAR run's NumPy int64 JSON crash."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run import write_results, make_report
from sihtigraph.walsh import fit_fourier, metrics, predict, random_masks


class ReceiptSerializationTests(unittest.TestCase):
    def test_randomly_sampled_branch_indices_are_python_ints(self):
        rng = np.random.default_rng(41)
        z = random_masks(rng, 256, 64)
        scores = z[:, 2] * z[:, 20] - .25 * z[:, 63]
        random_order_matched_subset = tuple(sorted(rng.choice(64, 3, replace=False)))
        self.assertIsInstance(random_order_matched_subset[0], np.integer)
        branches = fit_fourier(z, scores, [random_order_matched_subset], top=5)
        self.assertEqual(len(branches), 1)
        self.assertTrue(all(type(i) is int for i in branches[0]['indices']))
        json.dumps(branches, allow_nan=False)

    def test_full_report_receipt_accepts_numpy_dtypes(self):
        rng = np.random.default_rng(5)
        z = random_masks(rng, 300, 64)
        coeff_terms = fit_fourier(z, z[:, 1] * z[:, 7], [(np.int64(1), np.int64(7))])
        ztest = random_masks(rng, 200, 64)
        stats = metrics(ztest[:,1] * ztest[:,7], predict(ztest, coeff_terms, 0))
        scores = {name: dict(stats) for name in (
            'graph','linear','local_pair50','random_matched','intercept')}
        payload = {
            'image_index': np.int64(65),
            'class_name': 'bird',
            'teacher_correct': np.bool_(True),
            'checkpoint': {'validation_accuracy': np.float32(.7746)},
            'distillation': {
                'zero': {
                    'terms': [dict(coeff_terms[0], names=['B0:r0c1', 'B0:r1c3'])],
                    'km_query_count': np.int64(142336),
                    'teacher_forwards': np.int64(170000),
                    'teacher_calibration_sd': np.float64(2.1),
                    'evaluation': {'p=0.5':scores, 'p=0.75':scores, 'p=0.9':scores},
                    'mixed_difference': {'top_branch': np.array([1,7], dtype=np.int64)},
                }
            }
        }
        with tempfile.TemporaryDirectory() as dirname:
            root = Path(dirname)
            write_results(root/'results.json', payload)
            decoded=json.loads((root/'results.json').read_text(encoding='utf-8'))
            self.assertEqual(decoded['distillation']['zero']['mixed_difference']['top_branch'],[1,7])
            self.assertEqual(decoded['image_index'],65)
            self.assertEqual(decoded['distillation']['zero']['terms'][0]['indices'],[1,7])
            make_report(decoded,root/'report.html')
            report=(root/'report.html').read_text(encoding='utf-8')
            self.assertIn('bird',report)
            self.assertIn('Influence map',report)

    def test_nonfinite_receipt_rejected_without_replacing_old(self):
        with tempfile.TemporaryDirectory() as dirname:
            path=Path(dirname)/'results.json'
            path.write_text('{"good": true}',encoding='utf-8')
            with self.assertRaises(ValueError):
                write_results(path, {'bad': np.float64(np.nan)})
            self.assertEqual(path.read_text(encoding='utf-8'),'{"good": true}')


if __name__ == '__main__':
    unittest.main()
