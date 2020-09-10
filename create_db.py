from glob import glob
import cv2
import darknet
import os
import tensorflow as tf
import time
from utils import *

if __name__ == '__main__':
    raw_dir = './raw'
    db_dir = './database'
    # load yolo
    network, class_names, colors = darknet.load_network("yolo/fs_416.cfg", "yolo/obj.data", "yolo/yolo.weights")
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)
    
    #load landmark
    onet_graph = tf.Graph()
    with onet_graph.as_default():
        onet_graph_def = tf.GraphDef()
        with tf.gfile.GFile('onet.pb', 'rb') as f:
            onet_graph_def.ParseFromString(f.read())
            tf.import_graph_def(onet_graph_def)
        sess = tf.Session(graph=onet_graph)
    input_net = onet_graph.get_tensor_by_name('import/onet/input:0')
    output_net = onet_graph.get_tensor_by_name('import/onet/conv6-3/conv6-3:0')
    onet = lambda img: sess.run(output_net, feed_dict={input_net:img})
    img = cv2.imread('warmup.bmp')
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(rgb, (48, 48))
    tempimg = np.zeros((48, 48, 3, 1))
    tempimg[:, :, :, 0] = roi
    tempimg = np.transpose((tempimg - 127.5) * 0.0078125, (3, 1, 0, 2))
    print('Warm up landmark')
    output = onet(tempimg)
    dt = 0
    for i in range(5):
        t = time.time()
        output = onet(tempimg)
        dt += time.time() - t
    print('Avg', dt/5)
    
    for img_path in glob(os.path.join(raw_dir, '*.jpg')):
        name = os.path.basename(img_path)[:-4]
        print(name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, img.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5)
        if len(detections) > 0:
            #print(len(detections))
            dets = get_detections(detections, img, (width, height))
            # get first face only
            left, top, right, bottom, _ = [int(x) for x in dets[0]]
            roi = img[top:bottom, left:right]
            roi = cv2.resize(roi, (48, 48), cv2.INTER_AREA)
            tempimg = np.zeros((48, 48, 3, 1))
            tempimg[:, :, :, 0] = roi
            tempimg = np.transpose((tempimg - 127.5) * 0.0078125, (3, 1, 0, 2))
            output = onet(tempimg)
            w_bbox = right - left
            h_bbox = bottom - top
            x_anchor = left
            y_anchor = top
            x_coords = [int(x * w_bbox + x_anchor) for x in output[0][0:5]]
            y_coords = [int(y * h_bbox + y_anchor) for y in output[0][5:]]
            points = np.array(x_coords + y_coords).reshape((2, 5)).T
            #print(points)
            #pitch, yaw, roll = get_head_pose(img.shape, [left, top, right, bottom], points)
            #print(pitch, yaw, roll)
            #
            aligned_img = align(img, points)
            aligned_img = cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR)
            #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #cv2.rectangle(img, (left, top), (right, bottom), (255, 0, 0), 1)
            #for p in points:
            #    cv2.circle(img, (p[0], p[1]), 5, (0, 255, 0), 1)
            #cv2.imshow('abc', aligned_img)
            #cv2.waitKey(0)
            cv2.imwrite(os.path.join(db_dir, name + '.jpg'), aligned_img)
        else:
            print('No face found')
    print('Done')
    cv2.destroyAllWindows()
        
        
        
        
