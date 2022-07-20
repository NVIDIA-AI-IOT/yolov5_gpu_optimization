# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import ctypes
import tensorrt as trt
from trt_inference import (allocate_buffers, do_inference, do_inference_v2,
                           build_engine_from_onnx, load_tensorrt_engine, save_tensorrt_engine,
                           PTQEntropyCalibrator)
import numpy as np
import cv2
from functools import partial
import os
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool


INPUT_SIZE = 640
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']  # class names

COCO91_MAP = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
        35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
        64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

CONF_THRESH = 0.25


def preprocess_ds_nchw(batch_img):
    batch_img_array = np.array([np.array(img) for img in batch_img], dtype=np.float32)
    batch_img_array = batch_img_array / 255.0
    batch_transpose = np.transpose(batch_img_array, (0, 3, 1, 2))

    return batch_transpose

def preprocess_v2(batch_img):
    batch_img_array = np.zeros((len(batch_img),)+batch_img[0].shape, dtype=np.float32) 
    for idx, img in enumerate(batch_img):
        batch_img_array[idx] = img
    batch_img_array *= (1/255.0)
    batch_transpose = np.transpose(batch_img_array, (0, 3, 1, 2))

    return batch_transpose


def decode(keep_k, boxes, scores, cls_id):
    results = []
    for idx, k in enumerate(keep_k.reshape(-1)):
        bbox = boxes[idx].reshape((-1, 4))[:k]
        conf = scores[idx].reshape((-1, 1))[:k]
        cid = cls_id[idx].reshape((-1, 1))[:k]
        results.append(np.concatenate((cid, conf, bbox), axis=-1))
    
    return results


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def draw_bbox_cv(orig_img, infer_img, output_img_path, labels, ratio_pad=None, image_id=None, jlist=None):
    bboxes = labels[:, 2:]
    confs = labels[:, 1]
    cids = labels[:, 0]
    bboxes = scale_coords(infer_img.shape[2:], bboxes, orig_img.shape, ratio_pad=ratio_pad).round()
    
    for idx in range(len(labels)):
        
        bbox = bboxes[idx]
        p1, p2 = (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3]))
        cid = int(cids[idx])
        conf = confs[idx]
        # print("{}: {} {}".format(CLASSES[cid], conf, bbox))
        if jlist is not None:
            if image_id is not None:
                b = [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]]
                jlist.append({
                'image_id': image_id,
                'category_id': COCO91_MAP[cid],
                'bbox': [round(float(x), 3) for x in b],
                'score': round(float(conf), 5)})

        if conf < CONF_THRESH:
            continue

        cv2.rectangle(orig_img, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(orig_img, "{0}: {1:.2f}".format(CLASSES[cid], conf), p1, 0, 0.8, (255, 255, 0), 2)
    
    cv2.imwrite(output_img_path, orig_img)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # new_shape = (height, width)
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def load_images_cv(img_path, new_shape):
    orig_img = cv2.imread(img_path)
    img = letterbox(orig_img.copy(), new_shape, auto=False, scaleup=True)[0]
    img = img[..., [2, 1, 0]] # BGR -> RGB
    images = preprocess_ds_nchw([img])
    
    return images, orig_img

def load_images_cv_mt(img_paths, new_shape, pool):
    pass

def load_single_ul(img_path, img_size, new_shape):
    orig_img = cv2.imread(img_path)
    h0, w0 = orig_img.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA
        im = cv2.resize(orig_img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    else:
        im = orig_img
    img, _, _ = letterbox(im.copy(), new_shape, auto=False, scaleup=False)
    img = img[..., [2, 1, 0]] # BGR -> RGB
    return img

def load_images_cv_ultralytics_mt(img_paths, new_shape, pool, n_thread=8):

    load_func = partial(load_single_ul, img_size=INPUT_SIZE, new_shape=new_shape)
    cnt = 0
    imgs = []
    while cnt < len(img_paths):
        if cnt + n_thread >= len(img_paths):
            cur_map_list = img_paths[cnt :]
        else:
            cur_map_list = img_paths[cnt : cnt + n_thread]
        cnt += n_thread
        imgs.extend(pool.map(load_func, cur_map_list))

    images = preprocess_v2(imgs)
    return images

def load_images_cv_ultralytics(img_path, new_shape):
    orig_img = cv2.imread(img_path)
    h0, w0 = orig_img.shape[:2]  # orig hw
    r = INPUT_SIZE / max(h0, w0)  # ratio
    if r != 1:  # if sizes are not equal
        interp = cv2.INTER_LINEAR if (r > 1) else cv2.INTER_AREA
        im = cv2.resize(orig_img, (int(w0 * r), int(h0 * r)), interpolation=interp)
    else:
        im = orig_img
    h, w, _ = im.shape
    img, _, pad = letterbox(im.copy(), new_shape, auto=False, scaleup=False)
    ratio_pad = ((h / h0, w / w0), pad)  # for COCO mAP rescaling
    img = img[..., [2, 1, 0]] # BGR -> RGB
    images = preprocess_ds_nchw([img])
    
    return images, orig_img, ratio_pad


def rect_inference(engine, img_root, output_img_root, max_shape, img_new_shapes, jlist=None):
    with engine.create_execution_context() as context:
        context.set_binding_shape(0, max_shape)
        inputs, outputs, bindings, stream = allocate_buffers(engine, context)
        for img_name in tqdm(sorted(os.listdir(img_root))):
            img_path = os.path.join(img_root, img_name)
            if jlist is not None:
                img_id = int(img_name.split(".")[0])
            else:
                img_id = None
            
            new_shape = (INPUT_SIZE, INPUT_SIZE)
            context.set_optimization_profile_async(0, stream.handle)
            context.set_binding_shape(0, (1, 3, new_shape[0], new_shape[1]))
            images, orig_img, ratio_pad = load_images_cv_ultralytics(img_path, new_shape)

            batch_images = images
            # Hard Coded For explicit_batch and the ONNX model's batch_size = 1
            batch_images = batch_images[np.newaxis, :, :, :]
            outputs_shape, outputs_data = do_inference_v2(batch=batch_images, context=context,
                                                          bindings=bindings, inputs=inputs,
                                                          outputs=outputs, stream=stream)
            results = decode(keep_k = outputs_data["BatchedNMS"],
                             boxes = outputs_data["BatchedNMS_1"],
                             scores = outputs_data["BatchedNMS_2"],
                             cls_id = outputs_data["BatchedNMS_3"])
            # visualize the bbox
            draw_bbox_cv(orig_img, images, os.path.join(output_img_root, img_name),
                        results[0], image_id=img_id, jlist=jlist, ratio_pad=ratio_pad)


def square_inference(engine, img_root, output_img_root, jlist):
    with engine.create_execution_context() as context:
        context.set_binding_shape(0, (1, 3, INPUT_SIZE, INPUT_SIZE))
        new_shape = (INPUT_SIZE, INPUT_SIZE)
        inputs, outputs, bindings, stream = allocate_buffers(engine, context)
        for img_name in tqdm(sorted(os.listdir(img_root))):
            img_path = os.path.join(img_root, img_name)
            if jlist is not None:
                img_id = int(img_name.split(".")[0])
            else:
                img_id = None
            
            images, orig_img = load_images_cv(img_path, new_shape)
            ratio_pad = None
            batch_images = images
            # Hard Coded For explicit_batch and the ONNX model's batch_size = 1
            batch_images = batch_images[np.newaxis, :, :, :]
            outputs_shape, outputs_data = do_inference(batch=batch_images, context=context,
                                                        bindings=bindings, inputs=inputs,
                                                        outputs=outputs, stream=stream)
            results = decode(keep_k = outputs_data["BatchedNMS"],
                                boxes = outputs_data["BatchedNMS_1"],
                                scores = outputs_data["BatchedNMS_2"],
                                cls_id = outputs_data["BatchedNMS_3"])
            # visualize the bbox
            draw_bbox_cv(orig_img, images, os.path.join(output_img_root, img_name),
                         results[0], image_id=img_id, jlist=jlist, ratio_pad=ratio_pad)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Do YOLOV4 inference using TRT')
    parser.add_argument('--input_images_folder', type=str, help='input images path', required=True)
    parser.add_argument('--output_images_folder', type=str, help='output images path', required=True)
    parser.add_argument('--onnx', type=str, help='ONNX file path', required=True)
    parser.add_argument('--coco_anno', type=str, default="", help="COCO annotation file")
    parser.add_argument('--save_engine', type=str, default="", help="Save trt engine path")
    parser.add_argument('--rect', action="store_true", help="Do rect inference in COCO evaluation")
    parser.add_argument('--input_size', type=int, default=640, help="Input Size")
    parser.add_argument('--stride', type=int, default=32, help="the max stride of the model")
    parser.add_argument("--data_type", type=str, default="fp16", help="Data type for the TensorRT inference.", choices=["fp32", "fp16", "int8"])
    
    parser.add_argument('--calib_img_dir', type=str, default="", help="calibration images directory.")
    parser.add_argument('--calib_cache', type=str, default="", help="int8 calibration cache.")
    parser.add_argument('--n_batches', type=int, default=10, help="number of batches to do calibration.")
    parser.add_argument('--batch_size', type=int, default=1, help="batch size to do calibration.")

    args = parser.parse_args()

    batch_size = 1
    engine_file = args.onnx
    img_root = args.input_images_folder
    output_img_root = args.output_images_folder
    batch_cnt = 1
    INPUT_SIZE=args.input_size
    precision = args.data_type
    max_bs = args.batch_size
    save_engine = args.save_engine
    total_cnt = 0
    ac_cnt = 0
    
    if not os.path.exists(output_img_root):
        print("Please create the output images directory: {output_img_root}")
        exit(0)
    
    if args.coco_anno != "": # Do coco evaluation
        jlist = []
        # loop over the images to get the inference shape:
        if args.rect:
            img_new_shapes = {}
            pad = 0.5
            stride = args.stride
            min_w= 1e5
            min_h= 1e5
            max_w= -1
            max_h= -1
            # for img_name in sorted(os.listdir(img_root)):
            #     img_path = os.path.join(img_root, img_name)
            #     img = cv2.imread(img_path)
            #     h, w, _ = img.shape
            #     ar = h / w
            #     r = [1, 1]
            #     if ar < 1:
            #         r = [ar, 1]
            #     elif ar > 1:
            #         r = [1, 1/ar]
            #     new_shape = np.ceil(np.array(r) * INPUT_SIZE / stride + pad).astype(int) * stride
            #     if new_shape[0] < min_h:
            #         min_h = new_shape[0]
            #     elif new_shape[0] > max_h:
            #         max_h = new_shape[0]
            #     if new_shape[1] < min_w:
            #         min_w = new_shape[1]
            #     elif new_shape[1] > max_w:
            #         max_w = new_shape[1]
            #     img_new_shapes[img_name]=new_shape
            max_w, max_h = 672, 672
            INPUT_SIZE = max(max_w, max_h)
            min_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
            opt_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
            max_shape = (max_bs, 3, INPUT_SIZE, INPUT_SIZE)
            print(min_shape)
            print(opt_shape)
            print(max_shape)
        else:
            min_shape=(1, 3, INPUT_SIZE, INPUT_SIZE)
            opt_shape=(1, 3, INPUT_SIZE, INPUT_SIZE)
            max_shape=(max_bs, 3, INPUT_SIZE, INPUT_SIZE)
    else:
        jlist = None
        min_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
        opt_shape = (1, 3, INPUT_SIZE, INPUT_SIZE)
        max_shape = (max_bs, 3, INPUT_SIZE, INPUT_SIZE)
    
    if precision == "int8":
        n_thread = 8
        pool = Pool(n_thread)
        if args.rect:
            # load_func = partial(load_images_cv_ultralytics, new_shape=INPUT_SIZE)
            load_func = partial(load_images_cv_ultralytics_mt, new_shape=INPUT_SIZE, pool=pool, n_thread=n_thread)
        else:
            load_func = partial(load_images_cv, new_shape=INPUT_SIZE)

        calibrator = PTQEntropyCalibrator(cal_data=args.calib_img_dir,
                                          cache_file=args.calib_cache,
                                          load_func=load_func,
                                          n_batches=args.n_batches,
                                          batch_size=max_bs)
    else:
        calibrator = None

    trt.init_libnvinfer_plugins(None, '')
    # with load_tensorrt_engine(engine_file) as engine:
    #     print("Engine Loaded.")
    with build_engine_from_onnx(engine_file, verbose=False,
                                dtype=precision,
                                min_shape=min_shape,
                                opt_shape=opt_shape,
                                max_shape=max_shape,
                                extra_output_layer=[],
                                calibrator=calibrator
                               ) as engine:
        if save_engine != "":
            save_tensorrt_engine(save_engine, engine)
        if args.rect:
            rect_inference(engine, img_root=img_root, output_img_root=output_img_root,
                           max_shape=max_shape, img_new_shapes=img_new_shapes, jlist=jlist)
        else:
            square_inference(engine, img_root=img_root, output_img_root=output_img_root, jlist=jlist)

    if args.coco_anno != "":    
        anno_json = args.coco_anno  # annotations json
        pred_json = './cocoval17_predictions.json'  # predictions json
        import json
        with open(pred_json, 'w') as f:
            json.dump(jlist, f)

        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        anno = COCO(anno_json)  # init annotations api
        pred = anno.loadRes(pred_json)  # init predictions api
        eval = COCOeval(anno, pred, 'bbox')
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
