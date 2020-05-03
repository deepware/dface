import os, sys
import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import DBSCAN

from mtcnn import MTCNN
from facenet import FaceNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_frames(video):
	frames = []
	vid = cv2.VideoCapture(video)
	total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	nframe = total//30 # one frame in every 30 frames
	idx = np.linspace(0, total, nframe, endpoint=False, dtype=int)
	for i in range(total):
		ok = vid.grab()
		if i not in idx:
			continue
		ok, frm = vid.retrieve()
		if not ok:
			continue
		frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
		frames.append(frm)
	vid.release()
	return frames


def get_boundingbox(box, w, h, scale=1.2):
	x1, y1, x2, y2 = box
	size = int(max(x2-x1, y2-y1) * scale)
	center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
	if size > w or size > h:
		size = int(max(x2-x1, y2-y1))
	x1 = max(int(center_x - size // 2), 0)
	y1 = max(int(center_y - size // 2), 0)
	size = min(w - x1, size)
	size = min(h - y1, size)
	return x1, y1, size


def main():
	if len(sys.argv) != 2:
		print('usage: example.py <video>')
		return

	video = sys.argv[1]

	print("loading models.")

	mtcnn = MTCNN('models/mtcnn.pt', device)
	facenet = FaceNet('models/facenet.pt', device)

	print("reading video frames.")
	frames = get_frames(video)

	print("detecting & extracting faces.")
	result = mtcnn.detect(frames)

	faces = []
	for i, res in enumerate(result):
		if res is None:
			continue
		# extract faces
		boxes, probs, lands = res
		for j, box in enumerate(boxes):
			# confidence of detected face
			if probs[j] > 0.98:
				h, w = frames[i].shape[:2]
				x1, y1, size = get_boundingbox(box, w, h)
				face = frames[i][y1:y1+size, x1:x1+size]
				faces.append(face)

	print("creating face embeddings.")
	embeds = facenet.embedding(faces)

	print("clustering faces.")
	dbscan = DBSCAN(eps=0.35, metric='cosine', min_samples=5)
	labels = dbscan.fit_predict(embeds)

	name, _ = os.path.splitext(video)
	os.mkdir(name)

	print("saving clustered faces.")
	for i in range(len(labels)):
		label = labels[i]
		if label < 0:
			continue
		id_dir = '%s/id_%d'%(name, label)
		if not os.path.exists(id_dir):
			os.mkdir(id_dir)
		face = Image.fromarray(faces[i])
		face.save('%s/%d.bmp'%(id_dir, i))

main()