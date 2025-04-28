#include <gtest/gtest.h>

#include "ort_core/ort_core.h"
#include "tests/fps_counter.h"
#include "tests/image_drawer.h"
#include "detection_2d_util/detection_2d_util.h"
#include "sam2/sam2.hpp"

/**************************
****  ort core test ****
***************************/

using namespace inference_core;
using namespace detection_2d;
using namespace sam;

static std::shared_ptr<BaseSamModel> CreateModel()
{
  auto encoder_engine = CreateOrtInferCore("/workspace/models/image_encoder.engine");
  auto decoder_engine = CreateOrtInferCore("/workspace/models/image_decoder.engine");

  // auto preprocess_block = CreateCudaDetPreProcess();
  auto preprocess_block =
      CreateCpuDetPreProcess({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, true, true);

  auto model = sam::CreateSam2Model(encoder_engine, decoder_engine, preprocess_block);

  return model;
}

std::tuple<cv::Mat> ReadTestImage()
{
  auto image = cv::imread("/workspace/test_data/persons.jpg");
  CHECK(!image.empty());

  return {image};
}

TEST(sam2_test, trt_core_point_correctness)
{
  auto model   = CreateModel();
  auto [image] = ReadTestImage();

  cv::Mat masks;
  model->GenerateMask(image, {{225, 370}}, std::vector<int>{1}, masks, false);

  ImageDrawHelper helper(std::make_shared<cv::Mat>(image.clone()));
  helper.addRedMaskToForeground(masks);

  cv::imwrite("/workspace/test_data/sam2_result.png", *helper.getImage());
}
