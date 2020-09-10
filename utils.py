import numpy as np
import cv2
import darknet
import math

def get_detections(detections, image, source_shape):
    s_w, s_h = source_shape # source, eg 416x416
    t_h, t_w, _ = image.shape # target
    w_scale = float(t_w) / s_w
    h_scale = float(t_h) / s_h
    dets = []
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        x = x * w_scale
        y = y * h_scale
        w = w * w_scale
        h = h * h_scale
        left, top, right, bottom = darknet.bbox2points((x,y,w,h))
        dets.append([left, top, right, bottom, confidence])
    return np.asarray(dets, dtype=np.float32)
    
def get_head_pose(size, bbox, points):
    """
    :param size: img.shape
    :param bbox: [x1, y1, x2, y2]
    :param points: landmark reshape((2, 5)).T
    :return: pitch, yaw, roll: float
    """
    image_points = []
    image_points.append((points[2][0], points[2][1]))  # nose
    image_points.append(((points[3][0] + points[4][0]) / 2, bbox[3])) # chin
    image_points.append((points[0][0], points[0][1]))  # left eye
    image_points.append((points[1][0], points[1][1]))  # right eye
    image_points.append((points[3][0], points[3][1]))  # left mouth
    image_points.append((points[4][0], points[4][1]))  # right mouth

    image_points = np.asarray(image_points, dtype="double")
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corner
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])
    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs,
                                                                  flags=cv2.SOLVEPNP_ITERATIVE)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = cv2.hconcat((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return pitch, yaw, roll

def align(img, landmark):
    image_size = (112, 112)
    src = np.array([
        [30.2946, 51.6963],
        [65.5318, 51.5014],
        [48.0252, 71.7366],
        [33.5493, 92.3655],
        [62.7299, 92.2041]], dtype=np.float32)
    dst = landmark.astype(np.float32)
    M = cv2.estimateAffinePartial2D(dst, src)[0]
    warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
    return warped
