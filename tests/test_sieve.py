import unittest
try:
    import torch
    from sihtigraph.sieve import pieces, residues,render_mask
except ImportError:
    torch=None

@unittest.skipUnless(torch is not None,'torch optional for pure Fourier unit tests')
class SieveTests(unittest.TestCase):
    def test_lossless_bands_and_replacement(self):
        torch.manual_seed(41)
        x=torch.rand(1,3,32,32)
        self.assertLess(float((residues(x).sum(0,keepdim=True)-x).abs().max()),1e-5)
        for replacement in ('zero','mean'):
            base,part=pieces(x,replacement)
            self.assertEqual(tuple(part.shape),(64,3,32,32))
            all_on=torch.ones(1,64)
            self.assertLess(float((render_mask(all_on,base,part)-x).abs().max()),1e-5)
            self.assertLess(float((render_mask(-all_on,base,part)-base).abs().max()),1e-5)
            z=all_on.clone();z[0,19]=-1
            self.assertLess(float((x-render_mask(z,base,part)-part[19:20]).abs().max()),1e-5)

if __name__=='__main__':unittest.main()
