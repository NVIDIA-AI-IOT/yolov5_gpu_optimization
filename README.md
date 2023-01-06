# YOLOV5 inference solution in DeepStream and TensorRT
This repo provides sample codes to deploy YOLOV5 models in DeepStream or stand-alone TensorRT sample on Nvidia devices.

* [DeepStream sample](#deepstream-sample)
* [TensorRT sample](#tensorrt-sample)
* [Appendix](#appendix)

## DeepStream sample
In this section, we will walk through the steps to run YOLOV5 model using DeepStream with CPU NMS.
### Export the ultralytics YOLOV5 model to ONNX with TRT decode plugin
You could start from nvcr.io/nvidia/pytorch:22.03-py3 container for export.
```
git clone https://github.com/ultralytics/yolov5.git
# clone yolov5_trt_infer repo and copy the patch into yolov5 folder
git clone https://github.com/NVIDIA-AI-IOT/yolov5_gpu_optimization.git
cp yolov5_gpu_optimization/0001-Enable-onnx-export-with-decode-plugin.patch yolov5_gpu_optimization/requirement_export.txt yolov5/
cd yolov5
git checkout a80dd66efe0bc7fe3772f259260d5b7278aab42f
git am 0001-Enable-onnx-export-with-decode-plugin.patch
pip install -r requirement_export.txt
apt update && apt install -y libgl1-mesa-glx 
python export.py --weights yolov5s.pt --include onnx --simplify --dynamic
```
### Prepare the library for DeepStream inference.
You could start from nvcr.io/nvidia/deepstream:6.1.1-devel container for inference.

Then go to the deepstream sample directory.
```
cd deepstream-sample
```
Compile the plugin and deepstream parser:

* On x86:
    ```
    nvcc -Xcompiler -fPIC -shared -o yolov5_decode.so ./yoloForward_nc.cu ./yoloPlugins.cpp ./nvdsparsebbox_Yolo.cpp -isystem /usr/include/x86_64-linux-gnu/ -L /usr/lib/x86_64-linux-gnu/ -I /opt/nvidia/deepstream/deepstream/sources/includes -lnvinfer 
    ```
* On Jetson device:
    ```
    nvcc -Xcompiler -fPIC -shared -o yolov5_decode.so ./yoloForward_nc.cu ./yoloPlugins.cpp ./nvdsparsebbox_Yolo.cpp -isystem /usr/include/aarch64-linux-gnu/ -L /usr/lib/aarch64-linux-gnu/ -I /opt/nvidia/deepstream/deepstream/sources/includes -lnvinfer 
    ```
### Run inference
You could place the exported onnx models to `deepstream-sample`
```
cp yolov5/yolov5s.onnx yolov5_gpu_optimization/deepstream-sample/
```
Then you could run the model pre-defined configs.

* Run inference with saving inferened video:
    ```
    deepstream-app -c config/deepstream_app_config_save_video.txt 
    ```
* Run inference without display
    ```
    deepstream-app -c config/deepstream_app_config.txt 
    ```
* Run inference with 8 streams and batch_size=8 and without display
    ```
    deepstream-app -c config/deepstream_app_config_8s.txt 
    ```

### Performance summary:
The performance test is conducted on T4 with nvcr.io/nvidia/deepstream:6.1.1-devel

| Model   | Input Size | Device | precision | 1 stream bs=1 | 4 streams bs=4 | 8 streams bs=8 |
|---------|------------|--------|-----------|---------------|----------------|----------------|
| yolov5n | 3x640x640  | T4     | FP16      | 640           | 980            | 988            |
| yolov5m | 3x640x640  | T4     | FP16      | 220           | 270            | 277            |

## TensorRT sample
In this section, we will walk through the steps to run YOLOV5 model using GPU NMS with stand-alone inference script.
### Export the ultralytics YOLOV5 model to ONNX with TRT BatchNMS plugin
You could start from nvcr.io/nvidia/pytorch:22.03-py3 container for export.
```
git clone https://github.com/ultralytics/yolov5.git
# clone yolov5_trt_infer repo and copy files into yolov5 folder
git clone https://github.com/NVIDIA-AI-IOT/yolov5_gpu_optimization.git
cp -r yolov5_gpu_optimization/0001-Enable-onnx-export-with-batchNMS-plugin.patch yolov5_gpu_optimization/requirement_export.txt yolov5/
cd yolov5
git checkout a80dd66efe0bc7fe3772f259260d5b7278aab42f
git am 0001-Enable-onnx-export-with-batchNMS-plugin.patch
pip install -r requirement_export.txt
apt update && apt install -y libgl1-mesa-glx 
python export.py --weights yolov5s.pt --include onnx --simplify --dynamic
```

### Run with TensorRT:

For the following section, you could start from nvcr.io/nvidia/tensorrt:22.05-py3 and prepare env by:
```
cd tensorrt-sample
pip install -r requirement_infer.txt
apt update && apt install -y libgl1-mesa-glx 
```

Build plugin library by following the [previous steps](#prepare-the-library-for-deepstream-inference).
#### Run inference
```
python yolov5_trt_inference.py --input_images_folder=</path/to/coco/images/val2017/> --output_images_folder=./coco_output --onnx=</path/to/yolov5s.onnx>
```
#### Run evaluation on COCO17 validation dataset

##### Square inference evaluation:
The image will be resized to 3xINPUT_SIZExINPUT_SIZE while be kept aspect ratio.
```
python yolov5_trt_inference.py --input_images_folder=</path/to/coco/images/val2017/> --output_images_folder=<path/to/coco_output_dir> --onnx=</path/to/yolov5s.onnx> --coco_anno=</path/to/coco/annotations/instances_val2017.json> 
```

##### Rectangular inference evaluation:
This is not real rectangular inference as in pytorch. It is same to setting `pad=0, rect=False, imgsz=input_size + stride` in ultralytics YOLOV5.
```
# Default FP16 precision
python yolov5_trt_inference.py --input_images_folder=</path/to/coco/images/val2017/> --output_images_folder=<path/to/coco_output_dir> --onnx=</path/to/yolov5s.onnx> --coco_anno=</path/to/coco/annotations/instances_val2017.json> --rect
```


#### Eavaluation in INT8 mode
To run int8 inference or evaluation, you need to install TensorRT above 8.4. You could start from `nvcr.io/nvidia/tensorrt:22.07-py3`

Following command is to run evaluation in int8 precision (and calibration cache will be saved into the path specify by `--calib_cache`):
```
# INT8 precision
python yolov5_trt_inference.py --input_images_folder=</path/to/coco/images/val2017/> --output_images_folder=<path/to/coco_output_dir> --onnx=</path/to/yolov5s.onnx> --coco_anno=</path/to/coco/annotations/instances_val2017.json> --rect --data_type=int8 --save_engine=./yolov5s_int8_maxbs16.engine  --calib_img_dir=</path/to/coco/images/val2017/> --calib_cache=yolov5s_bs16_n10.cache --n_batches=10 --batch_size=16 
```

**Notes**: The calibration algorithm for YOLOV5 is `IInt8MinMaxCalibrator` instead of `IInt8EntropyCalibrator2`. So if you want to play with `trtexec` with the saved calibration cache, you have to change the first line of cache from `MinMaxCalibration` to `EntropyCalibration2`.

### Misc for TensorRT sample

#### Performance&&mAP summary
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

#### nbit-NMS
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
| 4              | Xavier NX | 10560          | 16         | 4          | 1.38                    |
| 4              | Xavier NX | 10560          | 8          | 4          | 1.08                    |

*Note*: small score bits may slightly decrease the final mAP. 

#### DeepStream deployment:
Users can intergrate the YOLOV5 with BatchedNMS plugin into DeepStream following [deepstream_tao_apps](https://github.com/NVIDIA-AI-IOT/deepstream_tao_apps)

## Appendix:
### YOLOV5 with different activation:
We conducted experiments with different activations for pursing better trade-off between mAP and performance on TensorRT.

You can change the activation of YOLOV5 model in `yolov5/models/common.py`:
```
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        # self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
```

YOLOV5s experiments results so far:

|     Activation type     |     mAP@0.5                                          |     V100 --best FPS (bs = 32)    |     A10  --best FPS (bs=32)    |
|-------------------------|------------------------------------------------------|----------------------------------|--------------------------------|
|     swish (baseline)    |     56.7%                                            |     1047                         |     965                        |
|     ReLU                |     54.8% (scratch)<br>55.7% (swish pretrained)      |     1177                         |     1065                       |
|     GELU                |     56.6%                                            |     1004                         |     916                        |
|     Leaky ReLU          |     55.0%                                            |     1172                         |     892                        |
|     PReLU               |     54.8%                                            |     1123                         |     932                        |

## Known issue:

- int8 0% mAP in TensorRT 8.2.5: Install TensorRT above 8.4 to avoid the issue.
- TensorRT warning at the end of the execution of stand-alone tensorrt inference script: The warning won't block the inference or evaluation. You can just ignore it.