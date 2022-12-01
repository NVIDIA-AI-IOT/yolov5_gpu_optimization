- Start from deepstream container:
```
nvidia-docker run -v <path>:<path>  --rm -it nvcr.io/nvidia/deepstream:6.1.1-devel bash
```

- Compile the lib:
```
nvcc -Xcompiler -fPIC -shared -o yolov5_decode.so ./yoloForward_nc.cu ./yoloPlugins.cpp ./nvdsparsebbox_Yolo.cpp -isystem /usr/include/x86_64-linux-gnu/ -L /usr/lib/x86_64-linux-gnu/ -I /opt/nvidia/deepstream/deepstream/sources/includes -lnvinfer 
```

- Run the deepstream sample:
```
deepstream -c config/deepstream_app_config.txt 
```


