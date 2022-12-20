import torch
import sys
from timeit import default_timer as timer

# load model
model = torch.hub.load('hustvl/yolop', 'yolop', pretrained=True)
model.eval()
start = timer()
model.to('cuda')

# inference
img = torch.randn(1, 3, 640, 640).to('cuda')
det_out, da_seg_out, ll_seg_out = model(img)

# d1 = det_out[0]
# d2 = da_seg_out[0]
# d3 = ll_seg_out[0]

print(det_out[0], file=sys.stderr)
print(da_seg_out[0], file=sys.stderr)
print(ll_seg_out[0], file=sys.stderr)

end = timer()
print(end - start)
