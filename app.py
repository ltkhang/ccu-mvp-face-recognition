import cv2
import time
import darknet
from sort import *
import numpy as np
import tensorflow as tf
import os
from glob import glob
from utils import *
from sklearn.preprocessing import normalize


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
    
    
def draw_boxes(detections, image, idx_dict):
    for bbox in detections:
        left, top, right, bottom, idx_track = [int(x) for x in bbox]
        name = ''
        if idx_track in idx_dict:
            name = idx_dict[idx_track]['name']
        cv2.rectangle(image, (left, top), (right, bottom), (255, 0, 0), 1)
        cv2.putText(image, "ID: {} {}".format(idx_track, name), (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

if __name__ == '__main__':
    db_dir = './database'
    MAX_FRAME = 30
    THRESHOLD = 1.0
    # load yolo
    network, class_names, colors = darknet.load_network("yolo/fs_192.cfg", "yolo/obj.data",
                                                "yolo/yolo.weights")
    vid_path = 0
    cap = cv2.VideoCapture(vid_path)
    vid_width = int(cap.get(3))
    vid_height = int(cap.get(4))
    vid_fps = cap.get(5)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    mot_tracker = Sort()
    # load face feature extraction model
    export_dir = "./tf131_model"
    sess = tf.Session(graph=tf.Graph())
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    x = sess.graph.get_tensor_by_name('data:0')
    y = sess.graph.get_tensor_by_name('fc1/add_1:0')
    fd = lambda img: sess.run(y, feed_dict={x: img})
    print('warm up face feature extraction')
    img = cv2.imread('warmup.bmp')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_data = np.asarray(img, np.float32)
    fd(np.expand_dims(input_data, axis=0))
    print('measure time of 10 iter')
    dt = 0
    for i in range(10):
        t = time.time()
        fd(np.expand_dims(input_data, axis=0))
        dt += time.time() - t
    print('Avg', dt / 10)
    # load landmark model
    #load landmark
    onet_graph = tf.Graph()
    with onet_graph.as_default():
        onet_graph_def = tf.GraphDef()
        with tf.gfile.GFile('onet.pb', 'rb') as f:
            onet_graph_def.ParseFromString(f.read())
            tf.import_graph_def(onet_graph_def)
        sess1 = tf.Session(graph=onet_graph)
    input_net = onet_graph.get_tensor_by_name('import/onet/input:0')
    output_net = onet_graph.get_tensor_by_name('import/onet/conv6-3/conv6-3:0')
    onet = lambda img: sess1.run(output_net, feed_dict={input_net:img})
    print('Warm up landmark')
    img = cv2.imread('warmup.bmp')
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(rgb, (48, 48))
    tempimg = np.zeros((48, 48, 3, 1))
    tempimg[:, :, :, 0] = roi
    tempimg = np.transpose((tempimg - 127.5) * 0.0078125, (3, 1, 0, 2))
    output = onet(tempimg)
    dt = 0
    for i in range(5):
        t = time.time()
        output = onet(tempimg)
        dt += time.time() - t
    print('Avg', dt/5)
    
    print('Start system')
    print('load DB')
    name_list = []
    features_list = []
    for img_path in glob(os.path.join(db_dir, '*.jpg')):
        name = os.path.basename(img_path)[:-4]
        print(img_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_data = np.asarray(img, np.float32)
        features = normalize(fd(np.expand_dims(input_data, axis=0))).flatten()
        name_list.append(name)
        features_list.append(features)
    features_list = np.array(features_list, dtype=np.float32)
    print('total id in db', len(name_list))
    idx_dict = dict()
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            t1 = time.time()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height),
                                       interpolation=cv2.INTER_LINEAR)
            darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5)
            dt = time.time() - t1
            #print(dt)
            #frame = cv2.putText(frame, 'Detection time: {:.4f}, fps: {}'.format(dt, 1 / dt), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, [255, 0, 0], 2)
            dets = get_detections(detections, frame, (width, height))
            track_bbs_ids = mot_tracker.update(dets)
            # cal landmark
            tempimg = np.zeros((48, 48, 3, len(track_bbs_ids)))
            idx_list = []
            coords = []
            for k in range(len(track_bbs_ids)):
                left, top, right, bottom, idx = [int(x) for x in track_bbs_ids[k]]
                roi = frame_rgb[top:bottom, left:right]
                if roi.shape[0] == 0 or roi.shape[1] == 0:
                    continue
                roi = cv2.resize(roi, (48, 48), cv2.INTER_AREA)
                tempimg[:, :, :, k] = roi
                if idx not in idx_dict:
                    idx_dict[idx] = dict()
                    idx_dict[idx]['roll'] = 1000
                    idx_dict[idx]['img'] = None
                    idx_dict[idx]['no_frame'] = 0 #number of tracked times
                    idx_dict[idx]['name'] = 'Unknown'
                    idx_dict[idx]['is_recognized'] = False
                if idx_dict[idx]['is_recognized'] == False:
                    idx_list.append(idx)
                    coords.append([left, top, right, bottom])
            tempimg = np.transpose((tempimg - 127.5) * 0.0078125, (3, 1, 0, 2))
            output = onet(tempimg)
            points = []
            for o in output:
                w_bbox = right - left
                h_bbox = bottom - top
                x_anchor = left
                y_anchor = top
                x_coords = [int(x * w_bbox + x_anchor) for x in o[0:5]]
                y_coords = [int(y * h_bbox + y_anchor) for y in o[5:]]
                p = np.array(x_coords + y_coords).reshape((2, 5)).T
                points.append(p)
            features_test = []
            idx_reg_list = []
            for i, idx in enumerate(idx_list):
                if idx_dict[idx]['no_frame'] < MAX_FRAME:
                    left, top, right, bottom = coords[i]
                    pitch, yaw, roll = get_head_pose(img.shape, [left, top, right, bottom], points[i])
                    if roll < idx_dict[idx]['roll']:
                        idx_dict[idx]['img'] = align(frame_rgb, points[i])
                        idx_dict[idx]['roll'] = roll
                    idx_dict[idx]['no_frame'] += 1
                else:
                     img = idx_dict[idx]['img']
                     input_data = np.asarray(img, np.float32)
                     features = normalize(fd(np.expand_dims(input_data, axis=0))).flatten()
                     idx_reg_list.append(idx)
                     features_test.append(features)
                     idx_dict[idx]['is_recognized'] = True
            if len(idx_reg_list) > 0:
                score_list = []
                for i, f1 in enumerate(features_test):
                    score_list.append([])
                    for f2 in features_list:
                        dist = np.sum(np.square(f1 - f2))
                        score_list[i].append(dist)
                for i, score in enumerate(score_list):
                    min_index = np.argmin(score)
                    idx = idx_reg_list[i]
                    if score[min_index] < THRESHOLD:
                        # person in database
                        name = name_list[min_index]
                        idx_dict[idx]['name'] = name
                    else:
                        print(score[min_index])
                        # person not in database, avoid missing classification, reset
                        idx_dict[idx]['roll'] = 1000
                        idx_dict[idx]['img'] = None
                        idx_dict[idx]['no_frame'] = 0 #number of tracked times
                        idx_dict[idx]['name'] = 'Unknown'
                        idx_dict[idx]['is_recognized'] = False
                
                    
            frame = draw_boxes(track_bbs_ids, frame, idx_dict)
            cv2.imshow('Detect {}x{}, Video fps {}'.format(vid_width, vid_height, vid_fps), frame)
            #out.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
