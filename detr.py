import math

from PIL import Image
import requests
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'

import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
from torch import nn
from torchvision.models import resnet50
import torchvision.transforms as transform
torch.set_grad_enabled(False)

classes = ['hammer', 'inverse hammer', 'bullish engulfing', 'piercing line', 'morningstar', 
           'three white soldiers', 'hanging man', 'shooting star', 'bearish engulfing', 
           'evening star', 'three black crows', 'dark cloud cover', 'doji', 'falling three', 'rising three']


transformations = transform.Compose([
    transform.Resize(800),
    transform.ToTensor(),
    transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.train()

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
im = Image.open(requests.get(url, stream=True).raw)


# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)

# propagate through the model
outputs = model(img)

# keep only predictions with 0.7+ confidence
probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
keep = probas.max(-1).values > 0.9