## dFace

Face detection and recognition library that focuses on speed and ease of use.
This is a stripped down version of timesler's [facenet](https://github.com/timesler/facenet-pytorch) repo with some improvements notably on memory overflows. This repository is actively developed and used in production.

[MTCNN](https://arxiv.org/abs/1604.02878) is used for face detection and [FaceNet](https://arxiv.org/abs/1503.03832) is used for generating face embeddings.

### Installation

Both algorithms can be used directly by copying the source file into your code base. They are single file and single model implementations with no interdependence.

Alternatively you can install with `pip3 install dface` command. Weights will be  downloaded from github automatically in the first run.

### MTCNN

This implementation takes a list of frames and returns a list of detected faces. The input and output lists always have the same size. If no face is detected for a frame, the corresponding list element is returned `None`, otherwise it's a tuple of three elements; `(bounding_boxes, probabilities, landmarks)`.

```python
mtcnn = MTCNN('cuda')
...
result = mtcnn.detect(frames)
for res in result:
    if res is None:
        continue
    boxes, probs, lands = res
```

Bounding boxes is a list of four coordinates `(x1, y1, x2, y2)` for each face detected in the frame. Probabilities is a list of confidence scores between 0 and 1 for each face. Landmarks is a list of facial landmark points. There are five points for each face.

The implementation is at [mtcnn.py](dface/mtcnn.py)

### FaceNet

This implementation takes a list of cropped faces and returns a list of face embedding vector of 512 dimensions. These embeddings then can be used for detecting facial similarities, clustering identities and so on.

```python
facenet = FaceNet('cuda')
...
embeds = facenet.embedding(faces)
```

The implementation is at [facenet.py](dface/facenet.py)

Both algorithms accept model path as the second argument if you want to load models from a custom location.

### Full working example

The repo contains an [example](example.py) script that uses both MTCNN and Facenet to read a video and cluster the faces. It uses [OpenCV](https://pypi.org/project/opencv-python/) to read video frames and [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html) to cluster faces.

You can run the example by simply typing: `python3 example.py video.mp4`. A directory will be created with the same name as video that contains clustered identities. The following is an example image.

![the dude](https://i.imgur.com/npt6W0l.png)

The implementation is at [example.py](example.py)
