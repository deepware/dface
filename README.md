## Facelib

Face detection and recognition library that focuses on speed and ease of use.
This is a stripped down version of timesler's [facenet](https://github.com/timesler/facenet-pytorch) repo with some improvements.

[MTCNN](https://arxiv.org/abs/1604.02878) is used for face detection and [FaceNet](https://arxiv.org/abs/1503.03832) is used for generating face embeddings.

This repo contains standalone pytorch implementations of both algorithms. They are single file and single model implementations with no interdependence.

### MTCNN

This implementation takes a list of frames and returns a list of detected faces. The input and output lists always have the same size. If no face is detected for a frame, the corresponding list element is `None`, otherwise it's a tuple of three elements; `(bounding_boxs, probabilities, landmarks)`.

```python
mtcnn = MTCNN('models/mtcnn.pt', 'cuda')
...
result = mtcnn.detect(frames)
for res in result:
    if res is None:
        continue
    boxes, probs, lands = res
```

Bounding boxes is a list of four coordinates `(x1, y1, x2, y2)` for each face detected in the frame. Probabilities is a list of confidence scores between 0 and 1 for each face. Landmarks is a list of facial landmark points. There are five points for each face.

### FaceNet

This implementation takes a list of cropped faces and returns a list of face embedding vector of 512 dimensions. These embeddings then can be used for detecting facial similarities, clustering identities and so on.

```python
facenet = FaceNet('models/facenet.pt', device)
...
embeds = facenet.embedding(faces)
```

### Full working example

The repo contains an example script that uses both MTCNN and Facenet to read a video and cluster the faces. It uses [OpenCV](https://pypi.org/project/opencv-python/) to read video frames and [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) to cluster faces.

You can run the example by simply typing: `python3 example.py video.mp4`. A directory will be created with the same name as video that contains clustered identities. The following is an example image.

![the dude](https://i.imgur.com/npt6W0l.png)

