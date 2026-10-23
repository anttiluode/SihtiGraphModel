"""Mathematical/regression checks for v3: no CIFAR or GPU needed."""
from __future__ import annotations
import tempfile
import unittest
from pathlib import Path
import numpy as np

from sihtigraph.v2 import graph_predict
from sihtigraph.v3 import (RETENTIONS,biased_coordinates,biased_degree12,
                           mixed_candidate_scores,choose_for_strata,
                           same_tile_cross_octave,compare_same_tile,synthetic_biased)
from sihtigraph.walsh import random_masks,metrics
from run_v3 import Cache,masks_for,render_html


class V3Tests(unittest.TestCase):
    def test_p_biased_moments_and_raw_joint_basis(self):
        result=synthetic_biased(41)
        self.assertAlmostEqual(result['unary_biased_2'],1.5,delta=.06)
        self.assertAlmostEqual(result['pair_biased_1_6'],-.8,delta=.06)
        self.assertGreater(result['heldout_r2'],.999)
        with self.assertRaises(ValueError):biased_coordinates(np.ones((1,3)),1.)

    def test_retention_mixture_recovers_a_pair_near_intact(self):
        rng=np.random.default_rng(45)
        def split(n):
            batches={}
            for p in RETENTIONS:
                z=random_masks(rng,n,8,p)
                y=1+.4*z[:,1]-1.2*z[:,1]*z[:,5]+.6*z[:,3]
                batches[p]=(z,y)
            return batches
        sel=split(3000);fit=split(900);val=split(500);test=split(1100)
        ranked=mixed_candidate_scores(sel)[:12]
        self.assertIn([1,5],[t['indices'] for t in ranked])
        graph,chosen=choose_for_strata(ranked,fit,val,budgets=(0,3,6,12))
        self.assertTrue(chosen['validation_by_count'])
        for p,(z,y) in test.items():
            self.assertGreater(metrics(y,graph_predict(z,graph))['r2'],.98)

    def test_no_spurious_pair_for_unary_at_p90(self):
        rng=np.random.default_rng(6)
        z=random_masks(rng,24000,10,.9);y=2*z[:,5]
        single,pair=biased_degree12(z,y,.9)
        self.assertGreater(abs(single[5]),.7)
        self.assertLess(abs(pair[5,6]),.15)

    def test_crossbird_tile_overlap_and_sign(self):
        a={'terms':[{'indices':[7,55],'coefficient':.3},
                    {'indices':[13,61],'coefficient':-.2},
                    {'indices':[5,12],'coefficient':.8}]}
        b={'terms':[{'indices':[13,61],'coefficient':-.1},
                    {'indices':[7,55],'coefficient':-.3},
                    {'indices':[4,52],'coefficient':.1}]}
        self.assertEqual(len(same_tile_cross_octave(a)),2)
        link=compare_same_tile(a,b)
        self.assertEqual(link['shared_tiles'],[7,13])
        self.assertEqual(link['same_sign_shared'],[13])

    def test_splits_cache_replay_and_corruption(self):
        for p in RETENTIONS:
            select=masks_for(41,65,'discover',p,132)
            self.assertFalse(np.array_equal(select,masks_for(41,65,'fit',p,132)))
            self.assertFalse(np.array_equal(select,masks_for(41,77,'discover',p,132)))
        self.assertTrue(np.all(masks_for(41,65,'all_on',1.,1)==1))
        class Oracle:
            def __init__(self):
                self.calls=0;self.clipped_pixels=0;self.total_pixels=0
            def __call__(self,z):
                self.calls+=len(z);self.clipped_pixels+=len(z)*3
                self.total_pixels+=len(z)*12
                return z[:,1].astype(float)
        with tempfile.TemporaryDirectory() as d:
            info={'version':3,'checkpoint_sha':'fixed','replacement':'mean'}
            first=Cache(Path(d),info);teacher=Oracle()
            z,y=first.read(41,65,'test',.9,200,teacher)
            restored=Cache(Path(d),info);again=Oracle()
            zz,yy=restored.read(41,65,'test',.9,200,again)
            self.assertEqual(again.calls,0)
            self.assertEqual(again.clipped_pixels,600)
            self.assertEqual(restored.loaded,200)
            self.assertTrue(np.array_equal(z,zz))
            self.assertTrue(np.array_equal(y,yy))
            path=next(first.path.glob('test_p090.npz'))
            with path.open('wb') as handle:
                np.savez_compressed(handle,z=np.zeros_like(z),y=y,clipped=0,total=0)
            with self.assertRaises(ValueError):
                restored.read(41,65,'test',.9,200,Oracle())

    def test_offline_report_mentions_unmasked_and_v2(self):
        metrics_at_p={'mixture_global':{'r2':.55},
                      'mixture_local':{'r2':.49},
                      'biased90_global':{'r2':.51},
                      'mixture_unary64':{'r2':.4},
                      'v2_frozen_global50':{'r2':-.4}}
        data={'images':{'65_mean':{
             'image_index':65,'replacement':'mean',
             'tests':{str(p):metrics_at_p for p in RETENTIONS},
             'full_image':{'teacher_margin':3.,'models':{'mixture_global':{'abs_margin_error':.2}}},
             'same_tile_B0_B3':[{'tile':7,'coefficient':.21}],
             'selection':{'mixture_global':{'chosen_count':50}},
             'oracle':{'fraction_pixels_clipped':0}
             }},'cross_bird':{'shared_tiles':[7],'same_sign_shared':[7]}}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'report.html'
            render_html(data,path)
            contents=path.read_text(encoding='utf8')
            for term in ('Fixed-p=.90','Frozen v2','0.95','Shared same-tile'):
                self.assertIn(term,contents)


if __name__=='__main__':unittest.main()
