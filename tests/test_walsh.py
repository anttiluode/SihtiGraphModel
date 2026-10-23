import unittest
import numpy as np
from sihtigraph.walsh import fwht, km_search, metrics, predict, random_masks, local_pairs

class WalshTests(unittest.TestCase):
    def test_exact_orthogonal_coefficients(self):
        n=6
        i=np.arange(2**n)
        z=1-2*((i[:,None]>>np.arange(n))&1)
        y=2. + 3*z[:,0] - 1.5*z[:,1]*z[:,3] + .5*z[:,0]*z[:,2]*z[:,5]
        th=fwht(y)
        self.assertAlmostEqual(th[0],2.)
        self.assertAlmostEqual(th[1],3.)
        self.assertAlmostEqual(th[(1<<1)|(1<<3)],-1.5)
        self.assertAlmostEqual(th[(1<<0)|(1<<2)|(1<<5)],.5)
        self.assertEqual(np.count_nonzero(np.abs(th)>1e-8),4)

    def test_km_finds_high_order(self):
        def f(z):return .2+1.5*z[:,0]-1.2*z[:,2]*z[:,4]+.9*z[:,1]*z[:,5]*z[:,8]
        x=random_masks(np.random.default_rng(123),4096,11)
        result=km_search(f,11,seed=5,tau=.2,q=1500,calibration_masks=4000,coefficient_masks=8192)
        found={tuple(t['indices']) for t in result['terms']}
        self.assertTrue({(0,),(2,4),(1,5,8)}.issubset(found),result['terms'])
        self.assertGreater(metrics(f(x),predict(x,result['terms'],result['intercept']))['r2'],.95)
        self.assertEqual(result['query_count']%2,0)

    def test_pairs_and_const(self):
        pairs=local_pairs()
        self.assertEqual(len(pairs),144)
        self.assertEqual(len(set(pairs)),144)
        self.assertEqual(len(set(i for p in pairs for i in p)),64)
        m=metrics(np.array([-1.,1.]),np.array([0.,0.]))
        self.assertEqual(m['r2'],0.)

if __name__=='__main__':unittest.main()
