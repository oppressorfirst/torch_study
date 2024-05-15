import torchvision.datasets

# train_data = torchvision.datasets.ImageNet("/dataset",split="train",transform=torchvision.transforms.ToTensor()
# ,download)

vgg16_false = torchvision.models.vgg16()
