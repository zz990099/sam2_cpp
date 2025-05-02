#!/bin/bash

/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/image_encoder.onnx \
                              --saveEngine=/workspace/models/image_encoder.engine \
                              --fp16

/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/image_decoder.onnx \
                              --saveEngine=/workspace/models/image_decoder.engine \
                              --fp16

/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/memory_encoder.onnx \
                              --saveEngine=/workspace/models/memory_encoder.engine \
                              --fp16

/usr/src/tensorrt/bin/trtexec --onnx=/workspace/models/memory_attention_opt.onnx \
                              --saveEngine=/workspace/models/memory_attention_opt.engine \
                              --minShapes=memory_0:1x256,memory_1:1x64x64x64,memory_pos_embed:4100x1x64 \
                              --optShapes=memory_0:16x256,memory_1:7x64x64x64,memory_pos_embed:28736x1x64 \
                              --maxShapes=memory_0:16x256,memory_1:7x64x64x64,memory_pos_embed:28736x1x64 \
                              --fp16
