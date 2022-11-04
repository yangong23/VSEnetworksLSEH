# ----------------------------------------------------------------
# Modified by Yan Gong
# Last revised: Feb 2022
# Reference: The orignal code is from VSRN: Visual Semantic Reasoning for Image-Text Matching (https://arxiv.org/pdf/1909.02701.pdf).
# The code has been modified from python2 to python3.
# -----------------------------------------------------------------

import torch
from vocab import Vocabulary
import evaluation_models

# for coco
print('Evaluation on COCO:')
evaluation_models.evalrank("/home/lunet/coyg4/multimodal/SCAN/Lsa_SCAN-master/runs/coco_scan/log/model_best.pth.tar", data_path='/home/lunet/coyg4/data/', split="test", fold5=False)

# for flickr
# print('Evaluation on Flickr30K:')
# # evaluation_models.evalrank("pretrain_model/flickr/model_fliker_1.pth.tar", "pretrain_model/flickr/model_fliker_2.pth.tar", data_path='/home/yan/data/', split="test", fold5=False)
# evaluation_models.evalrank("pretrain_model/flickr/model_fliker_LSA3.pth.tar", "pretrain_model/flickr/model_fliker_LSAN.pth.tar", data_path='/home/yan/data/', split="test", fold5=False)
#

