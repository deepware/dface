import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as TF


class FaceNet():
	def __init__(self, device=None, model=None):
		if device is None:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'

		resnet = InceptionResnetV1().to(device).eval()
		url = 'https://github.com/deepware/dFace/raw/master/models/facenet.pt'
		if model is None:
			resnet.load_state_dict(torch.hub.load_state_dict_from_url(url))
		else:
			resnet.load_state_dict(torch.load(model))
		self.model = resnet
		self.device = device

	def preprocess(self, faces):
		done = []
		for face in faces:
			if not isinstance(face, Image.Image):
				face = Image.fromarray(face)
			face = TF.resize(face, (160,160))
			face = TF.to_tensor(np.float32(face))
			face = (face - 127.5) / 128.0
			done.append(face)
		return torch.stack(done)

	def embedding(self, faces):
		if len(faces) == 0:
			return np.array([], np.float32)
		faces = self.preprocess(faces)
		embeds = []
		with torch.no_grad():
			for x in torch.split(faces, 40):
				y = self.model(x.to(self.device))
				embeds.append(y.cpu())
		return torch.cat(embeds).numpy()


class InceptionResnetV1(nn.Module):

	def __init__(self, device=None):
		super().__init__()

		# Define layers
		self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
		self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
		self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.maxpool_3a = nn.MaxPool2d(3, stride=2)
		self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
		self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
		self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
		self.repeat_1 = nn.Sequential(
			Block35(scale=0.17),
			Block35(scale=0.17),
			Block35(scale=0.17),
			Block35(scale=0.17),
			Block35(scale=0.17),
		)
		self.mixed_6a = Mixed_6a()
		self.repeat_2 = nn.Sequential(
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
			Block17(scale=0.10),
		)
		self.mixed_7a = Mixed_7a()
		self.repeat_3 = nn.Sequential(
			Block8(scale=0.20),
			Block8(scale=0.20),
			Block8(scale=0.20),
			Block8(scale=0.20),
			Block8(scale=0.20),
		)
		self.block8 = Block8(noReLU=True)
		self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
		self.dropout = nn.Dropout(0.6)
		self.last_linear = nn.Linear(1792, 512, bias=False)
		self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

	def forward(self, x):

		x = self.conv2d_1a(x)
		x = self.conv2d_2a(x)
		x = self.conv2d_2b(x)
		x = self.maxpool_3a(x)
		x = self.conv2d_3b(x)
		x = self.conv2d_4a(x)
		x = self.conv2d_4b(x)
		x = self.repeat_1(x)
		x = self.mixed_6a(x)
		x = self.repeat_2(x)
		x = self.mixed_7a(x)
		x = self.repeat_3(x)
		x = self.block8(x)
		x = self.avgpool_1a(x)
		x = self.dropout(x)
		x = self.last_linear(x.view(x.shape[0], -1))
		x = self.last_bn(x)
		x = F.normalize(x, p=2, dim=1)
		return x


class BasicConv2d(nn.Module):

	def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
		super().__init__()
		self.conv = nn.Conv2d(
			in_planes, out_planes,
			kernel_size=kernel_size, stride=stride,
			padding=padding, bias=False
		) # verify bias false
		self.bn = nn.BatchNorm2d(
			out_planes,
			eps=0.001, # value found in tensorflow
			momentum=0.1, # default pytorch value
			affine=True
		)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x = self.conv(x)
		x = self.bn(x)
		x = self.relu(x)
		return x


class Block35(nn.Module):

	def __init__(self, scale=1.0):
		super().__init__()

		self.scale = scale

		self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

		self.branch1 = nn.Sequential(
			BasicConv2d(256, 32, kernel_size=1, stride=1),
			BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
		)

		self.branch2 = nn.Sequential(
			BasicConv2d(256, 32, kernel_size=1, stride=1),
			BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
			BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
		)

		self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		out = torch.cat((x0, x1, x2), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		out = self.relu(out)
		return out


class Block17(nn.Module):

	def __init__(self, scale=1.0):
		super().__init__()

		self.scale = scale

		self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

		self.branch1 = nn.Sequential(
			BasicConv2d(896, 128, kernel_size=1, stride=1),
			BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
			BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
		)

		self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
		self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		out = self.relu(out)
		return out


class Block8(nn.Module):

	def __init__(self, scale=1.0, noReLU=False):
		super().__init__()

		self.scale = scale
		self.noReLU = noReLU

		self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

		self.branch1 = nn.Sequential(
			BasicConv2d(1792, 192, kernel_size=1, stride=1),
			BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
			BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
		)

		self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
		if not self.noReLU:
			self.relu = nn.ReLU(inplace=False)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		out = torch.cat((x0, x1), 1)
		out = self.conv2d(out)
		out = out * self.scale + x
		if not self.noReLU:
			out = self.relu(out)
		return out


class Mixed_6a(nn.Module):

	def __init__(self):
		super().__init__()

		self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

		self.branch1 = nn.Sequential(
			BasicConv2d(256, 192, kernel_size=1, stride=1),
			BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
			BasicConv2d(192, 256, kernel_size=3, stride=2)
		)

		self.branch2 = nn.MaxPool2d(3, stride=2)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		out = torch.cat((x0, x1, x2), 1)
		return out


class Mixed_7a(nn.Module):

	def __init__(self):
		super().__init__()

		self.branch0 = nn.Sequential(
			BasicConv2d(896, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 384, kernel_size=3, stride=2)
		)

		self.branch1 = nn.Sequential(
			BasicConv2d(896, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 256, kernel_size=3, stride=2)
		)

		self.branch2 = nn.Sequential(
			BasicConv2d(896, 256, kernel_size=1, stride=1),
			BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
			BasicConv2d(256, 256, kernel_size=3, stride=2)
		)

		self.branch3 = nn.MaxPool2d(3, stride=2)

	def forward(self, x):
		x0 = self.branch0(x)
		x1 = self.branch1(x)
		x2 = self.branch2(x)
		x3 = self.branch3(x)
		out = torch.cat((x0, x1, x2, x3), 1)
		return out