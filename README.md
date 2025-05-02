# sam2_cpp

## About This Project

该项目是`SAM2`算法的c++实现，包括`TensorRT`、`OnnxRuntime`三种硬件平台(推理引擎)，使用[SAM2Export](https://github.com/Aimol-l/SAM2Export)导出onnx模型，并对`MemoryAttention`模块进行了优化。

## Demo

| <img src="./assets/left.png" alt="1" width="500"> | <img src="./assets/disp_color.png" alt="1" width="500"> |
|:----------------------------------------:|:----:|
| **left image**  | **disp in color** |

以下带有**opt**标志的代表在模型导出工程[SAM2Export](https://github.com/Aimol-l/SAM2Export)基础上，优化`MemoryAttention`模型结构后导出的onnx模型，具体请查看[pr_link](https://github.com/Aimol-l/SAM2Export/pull/10).

|  jetson-orin-nx-16GB   |   qps   |  cpu   |
|:---------:|:---------:|:----------------:|
|  sam2_track(fp16) - origin   |   0.98   |  4%   |
|  sam2_track(fp16) - **opt**  |   **4.25**   |  17%   |

## Usage

### Download Project

下载git项目
```bash
git clone git@github.com:zz990099/sam2_cpp.git
cd sam2_cpp
git submodule init && git submodule update
```

### Build Enviroment

使用docker构建工作环境
```bash
cd sam2_cpp
bash easy_deploy_tool/docker/easy_deploy_startup.sh
# Select `jetson` -> `trt10_u2204`/`trt8_u2204`
bash easy_deploy_tool/docker/into_docker.sh
```

### Compile Codes

在docker容器内，编译工程. 使用 `-DENABLE_*`宏来启用某种推理框架，可用的有: `-DENABLE_TENSORRT=ON`、`-DENABLE_ORT=ON`，可以兼容。 
```bash
cd /workspace
mdkir build && cd build
cmake .. -DBUILD_TESTING=ON -DENABLE_TENSORRT=ON
make -j
```

### Convert Model

1. 从[google driver](https://drive.google.com/drive/folders/1EBDUN793q9mJwNC1NA5s2nfq0SxMTx4b?usp=drive_link)中下载模型，放到`/workspace/models/`下

2. 在docker容器内，运行模型转换脚本
```bash
cd /workspace
bash tools/cvt_onnx2trt.sh
```

### Run Test Cases

1. 下载测试数据[link](https://drive.google.com/drive/folders/13PwIl8TBYT54YhSAPmuKI98IB99GElOj?usp=drive_link)，放到`/workspace/test_data`下，文件夹名为`golf`

2. 运行测试用例，具体测试用例请参考代码。
```bash
cd /workspace/build
./bin/simple_tests --gtest_filter=*correctness
# 限制GLOG输出
GLOG_minloglevel=1 ./bin/simple_tests --gtest_filter=*track_speed
```

## References

- [sam2](https://github.com/facebookresearch/sam2)
- [SAM2Export](https://github.com/Aimol-l/SAM2Export)
- [EasyDeployTool](https://github.com/zz990099/EasyDeployTool)
