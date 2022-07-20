# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import os
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

logger = logging.getLogger(__name__)

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


# Array of TensorRT loggers. We need to keep global references to
# the TensorRT loggers that we create to prevent them from being
# garbage collected as those are referenced from C++ code without
# Python knowing about it.


tensorrt_loggers = []


def _create_tensorrt_logger(verbose=False):
    """Create a TensorRT logger.

    Args:
        verbose (bool): whether to make the logger verbose.
    """
    if verbose:
        # trt_verbosity = trt.Logger.Severity.INFO
        trt_verbosity = trt.Logger.Severity.VERBOSE
    else:
        trt_verbosity = trt.Logger.Severity.WARNING
    tensorrt_logger = trt.Logger(trt_verbosity)
    tensorrt_loggers.append(tensorrt_logger)
    return tensorrt_logger


# class PTQEntropyCalibrator(trt.IInt8LegacyCalibrator):
class PTQEntropyCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, cal_data, cache_file, load_func, n_batches, batch_size=64):
        # Whenever you specify a custom constructor for a TensorRT class,
        # you MUST call the constructor of the parent explicitly.
        trt.IInt8MinMaxCalibrator.__init__(self)

        self.cache_file = cache_file

        # Every time get_batch is called, the next batch of size batch_size will be copied to the device and returned.
        self.load_func = load_func
        self.img_list = [ os.path.join(cal_data, name) for name in os.listdir(cal_data) if name.split(".")[-1] in ["png", "jpg"] ]
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.current_index = 0

        self.device_input = None

    def get_batch_size(self):
        return self.batch_size

    # TensorRT passes along the names of the engine bindings to the get_batch function.
    # You don't necessarily have to use them, but they can be useful to understand the order of
    # the inputs. The bindings list is expected to have the same ordering as 'names'.
    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.img_list):
            return None
        
        current_batch = int(self.current_index / self.batch_size)
        if current_batch >= self.n_batches:
            return None
        
        cur_batch_img_paths = self.img_list[self.current_index : self.current_index + self.batch_size]
        batch = self.load_func(cur_batch_img_paths)
        
        if self.device_input is None:
            self.device_input = cuda.mem_alloc(
                batch.size * 4
            )  # 4 bytes per float32.

        if current_batch % 10 == 0:
            print("Calibrating batch {:}, containing {:} images".format(current_batch, self.batch_size))

        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(batch, dtype=np.float32))
        self.current_index += self.batch_size
        return [int(self.device_input)]

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
    # # Methods for LegacyCalibrator            
    # def get_quantile(self):
    #     return 1.0

    # def get_regression_cutoff(self):
    #     return 1.0

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, binding_name, shape=None):
        self.host = host_mem
        self.device = device_mem
        self.binding_name = binding_name
        self.shape = shape

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice\n" + str(self.device) + "Shape: " + str(self.shape)

    def __repr__(self):
        return self.__str__()

DEFAULT_MAX_WORKSPACE_SIZE = (1 << 30) * 8

def build_engine_from_onnx(
        onnx_filename,
        min_shape,
        opt_shape,
        max_shape,
        max_workspace_size=DEFAULT_MAX_WORKSPACE_SIZE,
        dtype="fp32",
        calibrator=None,
        fp32_layer_ids=None,
        fp16_layer_ids=None,
        layers_min_max_dict=None,
        verbose=False,
        layer_names=[],
        extra_output_layer=[],
        ):

        """Initialization routine."""
        if dtype == "int8":
            t_dtype = trt.DataType.INT8
        elif dtype == "fp16":
            t_dtype = trt.DataType.HALF
        elif dtype == "fp32":
            t_dtype = trt.DataType.FLOAT
        else:
            raise ValueError("Unsupported data type: %s" % dtype)

        if fp32_layer_ids is None:
            fp32_layer_ids = []
        elif dtype != "int8":
            raise ValueError(
                "FP32 layer precision could be set only when dtype is INT8"
            )

        if fp16_layer_ids is None:
            fp16_layer_ids = []
        elif dtype != "int8":
            raise ValueError(
                "FP16 layer precision could be set only when dtype is INT8"
            )


        tensorrt_logger = _create_tensorrt_logger(verbose)

        with trt.Builder(tensorrt_logger) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, tensorrt_logger) as parser:

            if t_dtype == trt.DataType.HALF and not builder.platform_has_fast_fp16:
                logger.error("Specified FP16 but not supported on platform.")
                raise AttributeError("Specified FP16 but not supported on platform.")
                return

            if t_dtype == trt.DataType.INT8 and not builder.platform_has_fast_int8:
                logger.error("Specified INT8 but not supported on platform.")
                raise AttributeError("Specified INT8 but not supported on platform.")
                return

            if t_dtype == trt.DataType.INT8 and calibrator is None and layers_min_max_dict is None:
                logger.error("Specified INT8 but no calibrator provided.")
                raise AttributeError("Specified INT8 but no calibrator provided.")


            with open(onnx_filename, 'rb') as model:
                if not parser.parse(model.read()):
                    print('ERROR: ONNX Parse Failed')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))


            # Save the layers names in prototxt:
            # for layer_idx in range(network.num_layers):
            #     layer = network.get_layer(layer_idx)
            #     layer_names.append(layer.name)
            #     print(f"{layer_idx} // {layer.name}:{layer.type}")

            for layer_idx in extra_output_layer:
                layer = network.get_layer(layer_idx)
                output_tensor = layer.get_output(0)
                network.mark_output(output_tensor)

            config = builder.create_builder_config()
            opt_profile = builder.create_optimization_profile()
            image_input = network.get_input(0)
            input_shape = image_input.shape
            input_name = image_input.name
            print("{}:{}".format(input_name, input_shape))
            opt_profile.set_shape(input="images",
                                  min=min_shape,
                                  opt=opt_shape,
                                  max=max_shape)
            config.add_optimization_profile(opt_profile)
            config.max_workspace_size = max_workspace_size

            if t_dtype == trt.DataType.HALF:
                print("Generating FP16 engine")
                config.flags |= 1 << int(trt.BuilderFlag.FP16)
                config.flags |= 1 << int(trt.BuilderFlag.STRICT_TYPES)

            if t_dtype == trt.DataType.INT8:
                print("Generating INT8 engine")
                config.flags |= 1 << int(trt.BuilderFlag.INT8)
                config.flags |= 1 <<int(trt.BuilderFlag.STRICT_TYPES)
                config.int8_calibrator = calibrator
                # # When use mixed precision, for TensorRT builder:
                # # strict_type_constraints needs to be True;
                # # fp16_mode needs to be True if any layer uses fp16 precision.
                # set_strict_types, set_fp16_mode = \
                #     _set_excluded_layer_precision(
                #         network=network,
                #         fp32_layer_names=self._fp32_layer_names,
                #         fp16_layer_names=self._fp16_layer_names,
                #     )
                # if set_strict_types:
                #     config.set_flag(trt.BuilderFlag.STRICT_TYPES)
                # if set_fp16_mode:
                #     config.set_flag(trt.BuilderFlag.FP16)

            engine = builder.build_engine(network, config)

            assert engine

            return engine


def allocate_buffers(engine, context):

    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        binding_id = engine.get_binding_index(str(binding))
        size = trt.volume(context.get_binding_shape(binding_id)) * engine.max_batch_size
        print("{}:{}".format(binding, size))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, binding))
        else:
            output_shape = engine.get_binding_shape(binding)
            if len(output_shape) == 3:
                dims = trt.Dims3(engine.get_binding_shape(binding))
                output_shape = (engine.max_batch_size, dims[0], dims[1], dims[2])
            elif len(output_shape) == 2:
                dims = trt.Dims2(output_shape)
                output_shape = (engine.max_batch_size, dims[0], dims[1])
            outputs.append(HostDeviceMem(host_mem, device_mem, binding, output_shape))

    return inputs, outputs, bindings, stream
    # return inputs, outputs, bindings



def do_inference(batch, context, bindings, inputs, outputs, stream):
    batch_size = batch.shape[0]
    assert len(inputs) == 1
    inputs[0].host = np.ascontiguousarray(batch, dtype=np.float32)
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()

    outputs_dict = {}
    outputs_shape = {}
    for out in outputs:
        outputs_dict[out.binding_name] = np.reshape(out.host, out.shape)
        outputs_shape[out.binding_name] = out.shape

    return outputs_shape, outputs_dict

def do_inference_v2(batch, context, bindings, inputs, outputs, stream):
    assert len(inputs) == 1
    inputs[0].host = np.ascontiguousarray(batch, dtype=np.float32)
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    stream.synchronize()
    outputs_dict = {}
    outputs_shape = {}
    for idx, out in enumerate(outputs):
        idx += 1
        output_shape = context.get_binding_shape(idx)
        output_size = np.array(output_shape).prod()
        outputs_dict[out.binding_name] = np.reshape(out.host[0:output_size], output_shape)
        outputs_shape[out.binding_name] = output_shape

    return outputs_shape, outputs_dict

def load_tensorrt_engine(filename, verbose=False):
    tensorrt_logger = _create_tensorrt_logger(verbose)

    if not os.path.exists(filename):
        raise ValueError("{} does not exits".format(filename))

    with trt.Runtime(tensorrt_logger) as runtime, open(filename, "rb") as f:
        trt_engine = runtime.deserialize_cuda_engine(f.read())

    return trt_engine

def save_tensorrt_engine(filename, trt_engine):
    with open(filename, "wb") as f:
        f.write(trt_engine.serialize())
