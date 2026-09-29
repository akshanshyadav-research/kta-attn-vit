import unittest
import numpy as np
import torch
from scipy.stats import kendalltau
from kta_attn.ranking import kendall_scores
from kta_attn.model import PatchDropout, CustomVisionTransformer

class CoreTests(unittest.TestCase):
    def test_notebook_equivalence(self):
        torch.manual_seed(5)
        cls=torch.randn(2,12);tokens=torch.randn(2,7,12)
        a=cls[:,None,:,None]-cls[:,None,None,:]
        b=tokens[:,:,:,None]-tokens[:,:,None,:]
        expected=torch.triu(a.sign()*b.sign(),diagonal=1).sum((-1,-2))/66
        torch.testing.assert_close(kendall_scores(cls,tokens,chunk_size=3),expected)

    def test_tau_b_with_ties(self):
        cls=torch.tensor([[1.,2.,2.,4.]])
        tokens=torch.tensor([[[1.,1.,3.,4.],[4.,3.,2.,1.],[0.,0.,0.,0.]]])
        actual=kendall_scores(cls,tokens,'tau-b')[0]
        for i in range(2):self.assertAlmostEqual(actual[i].item(),kendalltau(cls[0],tokens[0,i]).statistic,places=6)
        self.assertEqual(actual[2].item(),0)

    def test_selection_preserves_cls_and_full_retention(self):
        torch.manual_seed(3);x=torch.randn(1,197,8)
        torch.testing.assert_close(PatchDropout(1,'notebook')(x,True),x)
        y=PatchDropout(.5,'notebook')(x,True)
        torch.testing.assert_close(y[:,0],x[:,0]);self.assertGreaterEqual(y.shape[1],100)

    def test_model_forward_and_batch_guard(self):
        torch.set_num_threads(2)
        m=CustomVisionTransformer(img_size=224,patch_size=16,embed_dim=24,depth=2,
             num_heads=3,num_classes=10,patch_drop_blocks=[0],keep_ratio=.5,ats_ratio=.2).eval()
        with torch.inference_mode():y=m(torch.randn(1,3,224,224),mask=True)
        self.assertEqual(y.shape,(1,10));self.assertTrue(torch.isfinite(y).all())
        with self.assertRaises(ValueError):m(torch.randn(2,3,224,224),mask=True)

    def test_final_block_pruning_preserves_prediction(self):
        torch.manual_seed(12)
        m=CustomVisionTransformer(img_size=224,patch_size=16,embed_dim=24,depth=2,
            num_heads=3,num_classes=10,patch_drop_blocks=[],keep_ratio=.5).eval()
        m.kta_block=1
        x=torch.randn(1,3,224,224)
        with torch.inference_mode():
            torch.testing.assert_close(m(x),m(x,mask=True))

if __name__=='__main__':unittest.main()
