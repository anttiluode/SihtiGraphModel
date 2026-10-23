import unittest
try:
    import torch
    from sihtigraph.teacher import TinyCNN, ImageOracle, normalize
except ImportError:
    torch=None

@unittest.skipUnless(torch is not None,'torch unavailable')
class TeacherTests(unittest.TestCase):
    def test_image_oracle_matches_full_model_and_counts(self):
        torch.manual_seed(42)
        model=TinyCNN().eval()
        img=torch.rand(1,3,32,32)
        oracle=ImageOracle(model,img,2,'mean','cpu',batch=4)
        x=oracle(torch.ones((3,64),dtype=torch.int8).numpy())
        with torch.inference_mode():
            logits=model(normalize(img))
            other=logits.clone();other[:,2]=-1e9
            expect=float((logits[:,2]-other.max(1).values)[0])
        self.assertLess(abs(x[0]-expect),1e-5)
        self.assertTrue(all(abs(v-expect)<1e-5 for v in x))
        self.assertEqual(oracle.calls,3)
        self.assertLess(oracle.reconstruction_error,1e-5)

if __name__=='__main__':unittest.main()
