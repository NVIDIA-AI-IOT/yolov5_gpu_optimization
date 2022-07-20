# YOLOV5 TensorRT inference sample

## Export the ultralytics YOLOV5 model to ONNX with TRT BatchNMS plugin
You could start from nvcr.io/nvidia/pytorch:22.03-py3 container for export.
```
git clone https://github.com/ultralytics/yolov5.git
# clone yolov5_trt_infer repo and copy files into yolov5 folder
git clone https://github.com/NVIDIA-AI-IOT/yolov5_gpu_optimization.git
cp yolov5_gpu_optimization/* yolov5/
cd yolov5
git checkout a80dd66efe0bc7fe3772f259260d5b7278aab42f
git am 0001-Enable-onnx-export-with-batchNMS-plugin.patch
pip install -r requirement_export.txt
apt update && apt install -y libgl1-mesa-glx 
python export.py --weights yolov5s.pt --include onnx --simplify --dynamic
```

## Run with TensorRT:

For the following section, you could start from nvcr.io/nvidia/tensorrt:22.05-py3 and prepare env by:
```
cd yolov5
pip install -r requirement_infer.txt
apt update && apt install -y libgl1-mesa-glx 
```
### Run inference
```
python yolov5_trt_inference.py --input_images_folder=</path/to/coco/images/val2017/> --output_images_folder=./coco_output --onnx=</path/to/yolov5s.onnx>
```
### Run evaluation on COCO17 validation dataset

#### Square inference evaluation:
The image will be resized to 3xINPUT_SIZExINPUT_SIZE while be kept aspect ratio.
```
python yolov5_trt_inference.py --input_images_folder=</path/to/coco/images/val2017/> --output_images_folder=<path/to/coco_output_dir> --onnx=</path/to/yolov5s.onnx> --coco_anno=</path/to/coco/annotations/instances_val2017.json> 
```

#### Rectangular inference evaluation:
This is not real rectangular inference as in pytorch due to some dynmaic shape limitation in TensorRT. It is same to setting `pad=0, rect=False, imgsz=input_size + stride` in ultralytics YOLOV5.
```
# Default FP16 precision
python yolov5_trt_inference.py --input_images_folder=</path/to/coco/images/val2017/> --output_images_folder=<path/to/coco_output_dir> --onnx=</path/to/yolov5s.onnx> --coco_anno=</path/to/coco/annotations/instances_val2017.json> --rect
```


To run int8 inference or evaluation, you need to install TensorRT 8.4 in the container.
 - download the TensorRT 8.4 tar ball from [devzone](https://developer.nvidia.com/tensorrt)
   - Download the tar package. (e.g: [TensorRT 8.4 GA for Linux x86_64 and CUDA 11.0, 11.1, 11.2, 11.3, 11.4, 11.5, 11.6 and 11.7 TAR Package](https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/8.4.1/tars/tensorrt-8.4.1.5.linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz))
 - extract the files and install new tensorrt python wheel:
 ```
 tar zxvf TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz
 cd TensorRT-8.4.1.5/python
 pip install tensorrt-8.4.1.5-cp38-none-linux_x86_64.whl
 ```
 - set the LD_LIBRARY_PATH:
 ```
 export LD_LIBRARY_PATH=</path/to/TensorRT-8.4 lib>:$LD_LIBRARY_PATH
 ```

Following command is to run evaluation in int8 precision:
```
# INT8 precision
python yolov5_trt_inference.py --input_images_folder=</path/to/coco/images/val2017/> --output_images_folder=<path/to/coco_output_dir> --onnx=</path/to/yolov5s.onnx> --coco_anno=</path/to/coco/annotations/instances_val2017.json> --rect --data_type=int8 --save_engine=./yolov5s_int8_maxbs16.engine  --calib_img_dir=</path/to/coco/images/val2017/> --calib_cache=yolov5s_bs16_n10.cache --n_batches=10 --batch_size=16 
```

## Appendix

### Performance&&mAP summary
Here is the performance and mAP summary. Tested on V100 16G with TensorRT 8.2.5 in rectangular inference mode.

| Model    | Input Size | precision | FPS bs=32 | FPS bs= 1 | mAP@0.5 |
| -------- | ---------- | --------- | --------- | --------- | ------- |
| yolov5n  | 640        | FP16      | 1295      | 448       | 45.9%   |
| yolov5s  | 640        | FP16      | 917       | 378       | 57.1%   |
| yolov5m  | 640        | FP16      | 614       | 282       | 64%     |
| yolov5l  | 640        | FP16      | 416       | 202       | 67.3%   |
| yolov5x  | 640        | FP16      | 231       | 135       | 68.5%   |
| yolov5n6 | 1280       | FP16      | 341       | 160       | 54.2%   |
| yolov5s6 | 1280       | FP16      | 261       | 139       | 63.2%   |
| yolov5m6 | 1280       | FP16      | 155       | 99        | 68.8%   |
| yolov5l6 | 1280       | FP16      | 106       | 68        | 70.7%   |
| yolov5x6 | 1280       | FP16      | 60        | 45        | 71.9%   |

### nbit-NMS
Users can also enable nbit-NMS by changing the `scoreBits` in export.py. 
```python
# Default to be 16-bit
nms_attrs["scoreBits"] = 16
# Can be changed to smaller one to boost NMS operation:
# e.g. nms_attrs["scoreBits"] = 8
```
performance gain:
| Classes number | Device    | Anchors number | Score bits | Batch size | NMS Execution time (ms) |
| -------------- | --------- | -------------  | ---------- | ---------- | ----------------------- |
| 80             | A30       | 25200          | 16         | 32         | 12.1                    |
| 80             | A30       | 25200          | 8          | 32         | 10.0                    |
| 4              | Jetson NX | 10560          | 16         | 4          | 1.38                    |
| 4              | Jetson NX | 10560          | 8          | 4          | 1.08                    |

*Note*: small score bits may slightly decrease the final mAP. 

## Known issue:

- int8 0% mAP in TensorRT 8.2.5: Install TensorRT 8.4 to avoid the issue.
- Dynamic shape inference: The dynamic shape inference will run into CUDA error with some specific shapes. 
- TensorRT warning at the end of the execution: The warning won't block the inference or evaluation. You can just ignore it.