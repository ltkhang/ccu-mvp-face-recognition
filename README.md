# Face recognition on Jetson Nano (National Chung Cheng University - TEEP Intern project) 

## Resources

- Face detection: https://github.com/ltkhang/face-detection-yolov4-tiny
- Face landmark: MTCNN onet
- Face features: (mobilefacenet converted to  tensorflow) https://github.com/deepinsight/insightface
- SORT: https://github.com/abewley/sort

## Requirements

- tensorflow-gpu==1.3.1 (jetson nano's version)
- numpy
- scikit-learn
- filterpy

## Run

Put targets' image into directory 'raw'

run

```
python3 create_db.py
```

to extract face only of those targets (in 'database' directory, better create this directory first). These faces are resized and aligned to 112x112 (I still got a bug on align step, try to make it better later).

Get image from webcam as default

```
python3 app.py
```

## Workflow

We combined yolov4-tiny (3 yolo layers) at the size of 192x192 for face detection task, sort (simple online and realtime tracking) for face tracking task, and mobilefacenet (arcface) for face's features extraction task.

To observe the best threshold on Asian faces, we tested with 2 differnet Asian face dataset at repo: https://github.com/ltkhang/CASIA_FACE_V5_verification
