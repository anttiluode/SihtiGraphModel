import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from sihtigraph.v2 import (choose_graph, degree12_moments, degree3_moments,
                           fit_graph, graph_predict, local_allowed, ranked_candidates,
                           synthetic_40)
from sihtigraph.walsh import random_masks, r2
from run_v2 import StageCache, STAGES, stage_masks


class V2ReaderTests(unittest.TestCase):
    def test_40_distributed_additive_terms_early_and_late(self):
        # A v1 failure: 40 exactly representable terms vanish under KM pruning.
        for start in (0,24):
            with self.subTest(start=start):
                result=synthetic_40(seed=41,start=start)
                self.assertEqual(result['truth_found_in_top40'],40,result)
                self.assertEqual(result['max_pair_in_top40'],0,result)
                self.assertGreater(result['test_r2'],.999,result)

    def test_direct_pair_moments_and_heldout_fit(self):
        rng=np.random.default_rng(17)
        def oracle(z):return .3 + .9*z[:,0] - 1.3*z[:,4]*z[:,9] + 1.7*z[:,7]*z[:,13]
        select=random_masks(rng,12000,18)
        fit=random_masks(rng,5000,18)
        val=random_masks(rng,1200,18)
        test=random_masks(rng,1200,18)
        terms=ranked_candidates(degree12_moments(select,oracle(select)),4)
        self.assertIn((4,9),{tuple(x['indices']) for x in terms})
        self.assertIn((7,13),{tuple(x['indices']) for x in terms})
        model, info=choose_graph(terms,fit,oracle(fit),val,oracle(val),budgets=(0,1,2,3,4))
        self.assertGreater(r2(oracle(test),graph_predict(test,model)),.999)
        self.assertLessEqual(info['count'],4)

    def test_pure_triple_without_pairs(self):
        rng=np.random.default_rng(45)
        x=random_masks(rng,9000,12)
        def f(z):return z[:,1]*z[:,5]*z[:,10]
        low=ranked_candidates(degree12_moments(x,f(x)),5)
        self.assertLess(max(abs(i['score']) for i in low),.065)
        triples=ranked_candidates(degree3_moments(x,f(x),chunk=17),3)
        self.assertEqual(triples[0]['indices'],[1,5,10])
        self.assertGreater(abs(triples[0]['score']),.98)
        fit=random_masks(rng,1800,12);test=random_masks(rng,1000,12)
        model=fit_graph(fit,f(fit),triples[:1])
        self.assertGreater(r2(f(test),graph_predict(test,model)),.999)

    def test_fit_vs_validation_and_local_geometry(self):
        rng=np.random.default_rng(46)
        fit=random_masks(rng,1000,64)
        val=random_masks(rng,1000,64)
        self.assertFalse(np.array_equal(fit,val))
        allowed=local_allowed()
        self.assertIn((0,),allowed)
        self.assertIn((0,1),allowed)
        self.assertIn((0,16),allowed)
        self.assertNotIn((0,63),allowed)
        terms=[{'indices':[0],'score':.5}]
        model=fit_graph(fit,fit[:,0],terms)
        self.assertLess(max(abs(model['terms'][0]['coefficient']-1),abs(model['intercept'])),1e-5)

    def test_stage_cache_resume_and_input_binding(self):
        seed=41
        for stage in STAGES:
            z=stage_masks(seed,64,123,stage)
            self.assertEqual(z.shape,(123,64))
            self.assertTrue(np.isin(z,[-1,1]).all())
        self.assertFalse(np.array_equal(stage_masks(seed,64,123,'selection'),
                                        stage_masks(seed,64,123,'fit')))
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            identity={'seed':41,'checkpoint_sha256':'test-checkpoint','readout':'raw'}
            first=StageCache(root,identity)
            calls={'n':0}
            def oracle(z):
                calls['n']+=len(z)
                return z[:,0].astype(float)
            z,y=first.read('selection',123,seed,oracle)
            self.assertEqual(calls['n'],123)
            second=StageCache(root,identity)
            z2,y2=second.read('selection',123,seed,oracle)
            self.assertEqual(calls['n'],123)
            self.assertTrue(np.array_equal(z,z2))
            self.assertTrue(np.array_equal(y,y2))
            self.assertEqual(second.loaded,123)
            self.assertEqual(second.queried,0)
            # Altering the saved z would invalidate the cache rather than
            # silently contaminating a held-out or source-image experiment.
            cache_path=next(first.dir.glob('selection.npz'))
            with cache_path.open('wb') as f:
                np.savez_compressed(f,z=np.zeros_like(z),y=y)
            with self.assertRaises(ValueError):
                third=StageCache(root,identity)
                third.read('selection',123,seed,oracle)

    def test_v1_results_path_refused_by_runner(self):
        # The v2 script has no trainer and defaults to a new directory.
        from pathlib import Path
        s=Path('run_v2.py').read_text(encoding='utf8')
        self.assertNotIn('train_teacher(',s)
        self.assertIn("runs/v2_bird_seed41",s)
        self.assertIn("'v1_bird_seed41' in out.parts",s)


if __name__=='__main__':
    unittest.main()
