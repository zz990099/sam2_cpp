#include <gtest/gtest.h>

#include "trt_core/trt_core.h"
#include "tests/fps_counter.h"
#include "tests/image_drawer.h"
#include "tests/fs_util.h"
#include "detection_2d_util/detection_2d_util.h"
#include "sam2/sam2.hpp"

/**************************
****  trt core test ****
***************************/

using namespace inference_core;
using namespace detection_2d;
using namespace sam;

static std::shared_ptr<BaseSam2TrackModel> CreateModel()
{
  auto encoder_engine = CreateTrtInferCore("/workspace/models/image_encoder.engine");
  auto decoder_engine = CreateTrtInferCore("/workspace/models/image_decoder.engine");
  auto memory_attention_engine =
      CreateTrtInferCore("/workspace/models/memory_attention.engine",
                         {{"current_vision_feat", {1, 256, 64, 64}},
                          {"current_vision_pos_embed", {4096, 1, 256}},
                          {"memory_0", {16, 256}},
                          {"memory_1", {7, 64, 64, 64}},
                          {"memory_pos_embed", {28736, 1, 64}}},
                         {{"image_embed", {1, 256, 64, 64}}});
  auto memory_encoder_engine = CreateTrtInferCore("/workspace/models/memory_encoder.engine");

  // auto preprocess_block = CreateCudaDetPreProcess();
  auto preprocess_block =
      CreateCpuDetPreProcess({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, true, true);

  auto model = sam::CreateSam2Model(encoder_engine, decoder_engine, memory_attention_engine,
                                    memory_encoder_engine, preprocess_block);

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

TEST(sam2_test, trt_core_point_register_correctness)
{
  auto model   = CreateModel();
  auto [image] = ReadTestImage();

  cv::Mat masks;
  model->Register(image, {{225, 370}}, std::vector<int>{1}, masks, false);

  ImageDrawHelper helper(std::make_shared<cv::Mat>(image.clone()));
  helper.addRedMaskToForeground(masks);

  cv::imwrite("/workspace/test_data/sam2_register_result.png", *helper.getImage());
}

TEST(sam2_test, trt_core_point_track_correctness)
{
  auto model   = CreateModel();
  auto [image] = ReadTestImage();

  const std::filesystem::path demo_dir  = "/workspace/test_data/mustard0";
  auto                        rgb_paths = get_files_in_directory(demo_dir);
  std::sort(rgb_paths.begin(), rgb_paths.end());

  for (size_t i = 0; i < rgb_paths.size(); ++i)
  {
    LOG(INFO) << "cur rgb path : " << rgb_paths[i];
    cv::Mat image = cv::imread(rgb_paths[i].string());
    CHECK(!image.empty());

    cv::Mat masks;
    if (i == 0)
    {
      model->Register(image, {{130, 300}}, std::vector<int>{1}, masks, false);
    } else
    {
      model->Track(image, masks, false);
    }
    ImageDrawHelper helper(std::make_shared<cv::Mat>(image.clone()));
    helper.addRedMaskToForeground(masks);

    LOG(INFO) << "save path : " << std::string("/workspace/test_data/track_result/") / rgb_paths[i].filename();
    cv::imwrite(std::string("/workspace/test_data/track_result/") / rgb_paths[i].filename(),
                *helper.getImage());
  }
}