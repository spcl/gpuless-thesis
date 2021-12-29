import torch
from PIL import Image
from torchvision import transforms
from timeit import default_timer as timer

start = timer()

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext101_32x8d', pretrained=True)
# model = torch.load('resnext50_32x4d.pt')
model.eval()

input_image = Image.open('dog.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

input_batch = input_batch.to('cuda')
model.to('cuda')

with torch.no_grad():
    output = model(input_batch)

# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)

end = timer()
print(end - start)

# print(output[0])
# print(probabilities)

# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# with open("imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]

# top_prob, top_catid = torch.topk(probabilities, 5)
# print(categories[top_catid[0]], top_prob[0].item())

# # show top categories per image
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# for i in range(top5_prob.size(0)):
#     print(categories[top5_catid[i]], top5_prob[i].item())
