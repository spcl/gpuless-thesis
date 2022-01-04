import torch
import sys
from timeit import default_timer as timer

start = timer()

# load model
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
model.eval()
model.to('cuda')

#inference
img = torch.randn(1,3,640,640).to('cuda')
det_out, da_seg_out, ll_seg_out = model(img)
print(det_out[0], file=sys.stderr)
print(da_seg_out[0], file=sys.stderr)
print(ll_seg_out[0], file=sys.stderr)


end = timer()
print(end - start)
