import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision.ops.boxes import batched_nms


class MTCNN():
	def __init__(self, device=None, model=None):
		if device is None:
			device = 'cuda' if torch.cuda.is_available() else 'cpu'
		self.device = device

		url = 'https://github.com/deepware/dFace/raw/master/models/mtcnn.pt'
		if model is None:
			model = torch.hub.load_state_dict_from_url(url)
		else:
			model = torch.load(model, map_location=device)

		self.pnet = PNet().to(device)
		self.rnet = RNet().to(device)
		self.onet = ONet().to(device)

		self.pnet.load_state_dict(model['pnet'])
		self.rnet.load_state_dict(model['rnet'])
		self.onet.load_state_dict(model['onet'])


	def detect(self, imgs, minsize=None):
		if len(imgs) == 0:
			return []

		if isinstance(imgs[0], np.ndarray):
			h, w = imgs[0].shape[:2]
		else:
			w, h = imgs[0].size

		if minsize is None:
			minsize = max(96 * min(w, h)/1080, 40)

		boxes, points = [], []

		with torch.no_grad():
			batches = [imgs[i:i+10] for i in range(0, len(imgs), 10)]
			for batch in batches:
				batch_boxes, batch_points = detect_face(
					batch, minsize, self.pnet, self.rnet, self.onet,
					[0.7, 0.8, 0.9], 0.709, self.device)
				boxes += list(batch_boxes)
				points += list(batch_points)

		result = []
		for box, point in zip(boxes, points):
			box = np.array(box)
			point = np.array(point)
			if len(box) == 0:
				result.append(None)
			else:
				result.append((box[:, :4], box[:, 4], point))
		return result


def empty_cache(device):
	if 'cuda' in device:
		with torch.cuda.device(device):
			torch.cuda.empty_cache()


class PNet(nn.Module):

	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
		self.prelu1 = nn.PReLU(10)
		self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
		self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
		self.prelu2 = nn.PReLU(16)
		self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
		self.prelu3 = nn.PReLU(32)
		self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
		self.softmax4_1 = nn.Softmax(dim=1)
		self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.prelu1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.prelu2(x)
		x = self.conv3(x)
		x = self.prelu3(x)
		a = self.conv4_1(x)
		a = self.softmax4_1(a)
		b = self.conv4_2(x)
		return b, a


class RNet(nn.Module):

	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
		self.prelu1 = nn.PReLU(28)
		self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
		self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
		self.prelu2 = nn.PReLU(48)
		self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
		self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
		self.prelu3 = nn.PReLU(64)
		self.dense4 = nn.Linear(576, 128)
		self.prelu4 = nn.PReLU(128)
		self.dense5_1 = nn.Linear(128, 2)
		self.softmax5_1 = nn.Softmax(dim=1)
		self.dense5_2 = nn.Linear(128, 4)

	def forward(self, x):
		x = self.conv1(x)
		x = self.prelu1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.prelu2(x)
		x = self.pool2(x)
		x = self.conv3(x)
		x = self.prelu3(x)
		x = x.permute(0, 3, 2, 1).contiguous()
		x = self.dense4(x.view(x.shape[0], -1))
		x = self.prelu4(x)
		a = self.dense5_1(x)
		a = self.softmax5_1(a)
		b = self.dense5_2(x)
		return b, a


class ONet(nn.Module):

	def __init__(self):
		super().__init__()

		self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
		self.prelu1 = nn.PReLU(32)
		self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
		self.prelu2 = nn.PReLU(64)
		self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
		self.prelu3 = nn.PReLU(64)
		self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
		self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
		self.prelu4 = nn.PReLU(128)
		self.dense5 = nn.Linear(1152, 256)
		self.prelu5 = nn.PReLU(256)
		self.dense6_1 = nn.Linear(256, 2)
		self.softmax6_1 = nn.Softmax(dim=1)
		self.dense6_2 = nn.Linear(256, 4)
		self.dense6_3 = nn.Linear(256, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = self.prelu1(x)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.prelu2(x)
		x = self.pool2(x)
		x = self.conv3(x)
		x = self.prelu3(x)
		x = self.pool3(x)
		x = self.conv4(x)
		x = self.prelu4(x)
		x = x.permute(0, 3, 2, 1).contiguous()
		x = self.dense5(x.view(x.shape[0], -1))
		x = self.prelu5(x)
		a = self.dense6_1(x)
		a = self.softmax6_1(a)
		b = self.dense6_2(x)
		c = self.dense6_3(x)
		return b, c, a


def detect_face(imgs, minsize, pnet, rnet, onet, threshold, factor, device):
	if isinstance(imgs, (np.ndarray, torch.Tensor)):
		imgs = torch.as_tensor(imgs, device=device)
		if len(imgs.shape) == 3:
			imgs = imgs.unsqueeze(0)
	else:
		if not isinstance(imgs, (list, tuple)):
			imgs = [imgs]
		if any(img.size != imgs[0].size for img in imgs):
			raise Exception("MTCNN batch processing only compatible with equal-dimension images.")
		imgs = np.stack([np.uint8(img) for img in imgs])

	imgs = torch.as_tensor(imgs, device=device)

	model_dtype = next(pnet.parameters()).dtype
	imgs = imgs.permute(0, 3, 1, 2).type(model_dtype)

	batch_size = len(imgs)
	h, w = imgs.shape[2:4]
	m = 12.0 / minsize
	minl = min(h, w)
	minl = minl * m

	# Create scale pyramid
	scale_i = m
	scales = []
	while minl >= 12:
		scales.append(scale_i)
		scale_i = scale_i * factor
		minl = minl * factor

	# First stage
	boxes = []
	image_inds = []
	all_inds = []
	all_i = 0
	for scale in scales:
		im_data = imresample(imgs, (int(h * scale + 1), int(w * scale + 1)))
		im_data = (im_data - 127.5) * 0.0078125
		reg, probs = pnet(im_data)
		empty_cache(device)
		boxes_scale, image_inds_scale = generateBoundingBox(reg, probs[:, 1], scale, threshold[0])
		boxes.append(boxes_scale)
		image_inds.append(image_inds_scale)
		all_inds.append(all_i + image_inds_scale)
		all_i += batch_size

	boxes = torch.cat(boxes, dim=0)
	image_inds = torch.cat(image_inds, dim=0).cpu()
	all_inds = torch.cat(all_inds, dim=0)

	# NMS within each scale + image
	pick = batched_nms(boxes[:, :4], boxes[:, 4], all_inds, 0.5)
	boxes, image_inds = boxes[pick], image_inds[pick]

	# NMS within each image
	pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
	boxes, image_inds = boxes[pick], image_inds[pick]

	regw = boxes[:, 2] - boxes[:, 0]
	regh = boxes[:, 3] - boxes[:, 1]
	qq1 = boxes[:, 0] + boxes[:, 5] * regw
	qq2 = boxes[:, 1] + boxes[:, 6] * regh
	qq3 = boxes[:, 2] + boxes[:, 7] * regw
	qq4 = boxes[:, 3] + boxes[:, 8] * regh
	boxes = torch.stack([qq1, qq2, qq3, qq4, boxes[:, 4]]).permute(1, 0)
	boxes = rerec(boxes)
	y, ey, x, ex = pad(boxes, w, h)

	# Second stage
	if len(boxes) > 0:
		im_data = []
		for k in range(len(y)):
			if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
				img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
				im_data.append(imresample(img_k, (24, 24)))
		im_data = torch.cat(im_data, dim=0)
		im_data = (im_data - 127.5) * 0.0078125

		out = []
		for batch in im_data.split(2000):
			out += [rnet(batch)]
		z = list(zip(*out))
		out = (torch.cat(z[0]), torch.cat(z[1]))
		empty_cache(device)

		out0 = out[0].permute(1, 0)
		out1 = out[1].permute(1, 0)
		score = out1[1, :]
		ipass = score > threshold[1]
		boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
		image_inds = image_inds[ipass]
		mv = out0[:, ipass].permute(1, 0)

		# NMS within each image
		pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
		boxes, image_inds, mv = boxes[pick], image_inds[pick], mv[pick]
		boxes = bbreg(boxes, mv)
		boxes = rerec(boxes)

	# Third stage
	points = torch.zeros(0, 5, 2, device=device)
	if len(boxes) > 0:
		y, ey, x, ex = pad(boxes, w, h)
		im_data = []
		for k in range(len(y)):
			if ey[k] > (y[k] - 1) and ex[k] > (x[k] - 1):
				img_k = imgs[image_inds[k], :, (y[k] - 1):ey[k], (x[k] - 1):ex[k]].unsqueeze(0)
				im_data.append(imresample(img_k, (48, 48)))
		im_data = torch.cat(im_data, dim=0)
		im_data = (im_data - 127.5) * 0.0078125

		out = []
		for batch in im_data.split(500):
			out += [onet(batch)]
		z = list(zip(*out))
		out = (torch.cat(z[0]), torch.cat(z[1]), torch.cat(z[2]))
		empty_cache(device)

		out0 = out[0].permute(1, 0)
		out1 = out[1].permute(1, 0)
		out2 = out[2].permute(1, 0)
		score = out2[1, :]
		points = out1
		ipass = score > threshold[2]
		points = points[:, ipass]
		boxes = torch.cat((boxes[ipass, :4], score[ipass].unsqueeze(1)), dim=1)
		image_inds = image_inds[ipass]
		mv = out0[:, ipass].permute(1, 0)

		w_i = boxes[:, 2] - boxes[:, 0] + 1
		h_i = boxes[:, 3] - boxes[:, 1] + 1
		points_x = w_i.repeat(5, 1) * points[:5, :] + boxes[:, 0].repeat(5, 1) - 1
		points_y = h_i.repeat(5, 1) * points[5:10, :] + boxes[:, 1].repeat(5, 1) - 1
		points = torch.stack((points_x, points_y)).permute(2, 1, 0)
		boxes = bbreg(boxes, mv)

		# NMS within each image using "Min" strategy
		# pick = batched_nms(boxes[:, :4], boxes[:, 4], image_inds, 0.7)
		pick = batched_nms_numpy(boxes[:, :4], boxes[:, 4], image_inds, 0.7, 'Min')
		boxes, image_inds, points = boxes[pick], image_inds[pick], points[pick]

	boxes = boxes.cpu().numpy()
	points = points.cpu().numpy()

	batch_boxes = []
	batch_points = []
	for b_i in range(batch_size):
		b_i_inds = np.where(image_inds == b_i)
		batch_boxes.append(boxes[b_i_inds].copy())
		batch_points.append(points[b_i_inds].copy())

	batch_boxes, batch_points = np.array(batch_boxes), np.array(batch_points)
	empty_cache(device)

	return batch_boxes, batch_points


def bbreg(boundingbox, reg):
	if reg.shape[1] == 1:
		reg = torch.reshape(reg, (reg.shape[2], reg.shape[3]))

	w = boundingbox[:, 2] - boundingbox[:, 0] + 1
	h = boundingbox[:, 3] - boundingbox[:, 1] + 1
	b1 = boundingbox[:, 0] + reg[:, 0] * w
	b2 = boundingbox[:, 1] + reg[:, 1] * h
	b3 = boundingbox[:, 2] + reg[:, 2] * w
	b4 = boundingbox[:, 3] + reg[:, 3] * h
	boundingbox[:, :4] = torch.stack([b1, b2, b3, b4]).permute(1, 0)

	return boundingbox


def generateBoundingBox(reg, probs, scale, thresh):
	stride = 2
	cellsize = 12

	reg = reg.permute(1, 0, 2, 3)

	mask = probs >= thresh
	mask_inds = mask.nonzero()
	image_inds = mask_inds[:, 0]
	score = probs[mask]
	reg = reg[:, mask].permute(1, 0)
	bb = mask_inds[:, 1:].type(reg.dtype).flip(1)
	q1 = ((stride * bb + 1) / scale).floor()
	q2 = ((stride * bb + cellsize - 1 + 1) / scale).floor()
	boundingbox = torch.cat([q1, q2, score.unsqueeze(1), reg], dim=1)
	return boundingbox, image_inds


def nms_numpy(boxes, scores, threshold, method):
	if boxes.size == 0:
		return np.empty((0, 3))

	x1 = boxes[:, 0].copy()
	y1 = boxes[:, 1].copy()
	x2 = boxes[:, 2].copy()
	y2 = boxes[:, 3].copy()
	s = scores
	area = (x2 - x1 + 1) * (y2 - y1 + 1)

	I = np.argsort(s)
	pick = np.zeros_like(s, dtype=np.int16)
	counter = 0
	while I.size > 0:
		i = I[-1]
		pick[counter] = i
		counter += 1
		idx = I[0:-1]

		xx1 = np.maximum(x1[i], x1[idx]).copy()
		yy1 = np.maximum(y1[i], y1[idx]).copy()
		xx2 = np.minimum(x2[i], x2[idx]).copy()
		yy2 = np.minimum(y2[i], y2[idx]).copy()

		w = np.maximum(0.0, xx2 - xx1 + 1).copy()
		h = np.maximum(0.0, yy2 - yy1 + 1).copy()

		inter = w * h
		if method == "Min":
			o = inter / np.minimum(area[i], area[idx])
		else:
			o = inter / (area[i] + area[idx] - inter)
		I = I[np.where(o <= threshold)]

	pick = pick[:counter].copy()
	return pick


def batched_nms_numpy(boxes, scores, idxs, threshold, method):
	device = boxes.device
	if boxes.numel() == 0:
		return torch.empty((0,), dtype=torch.int64, device=device)
	# strategy: in order to perform NMS independently per class.
	# we add an offset to all the boxes. The offset is dependent
	# only on the class idx, and is large enough so that boxes
	# from different classes do not overlap
	max_coordinate = boxes.max()
	offsets = idxs.to(boxes) * (max_coordinate + 1)
	boxes_for_nms = boxes + offsets[:, None]
	boxes_for_nms = boxes_for_nms.cpu().numpy()
	scores = scores.cpu().numpy()
	keep = nms_numpy(boxes_for_nms, scores, threshold, method)
	return torch.as_tensor(keep, dtype=torch.long, device=device)


def pad(boxes, w, h):
	boxes = boxes.trunc().int().cpu().numpy()
	x = boxes[:, 0]
	y = boxes[:, 1]
	ex = boxes[:, 2]
	ey = boxes[:, 3]

	x[x < 1] = 1
	y[y < 1] = 1
	ex[ex > w] = w
	ey[ey > h] = h

	return y, ey, x, ex


def rerec(bboxA):
	h = bboxA[:, 3] - bboxA[:, 1]
	w = bboxA[:, 2] - bboxA[:, 0]

	l = torch.max(w, h)
	bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
	bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
	bboxA[:, 2:4] = bboxA[:, :2] + l.repeat(2, 1).permute(1, 0)

	return bboxA


def imresample(img, sz):
	im_data = interpolate(img, size=sz, mode="area")
	return im_data