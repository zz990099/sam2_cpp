#include <gtest/gtest.h>

#include "fps_counter.h"
#include "image_drawer.h"
#include "fs_util.h"
#include "detection_2d_util/detection_2d_util.h"
#include "sam2/sam2.hpp"

/**************************
****  trt core test ****
***************************/

using namespace inference_core;
using namespace detection_2d;
using namespace sam;

void sam2_test_correctness(const std::shared_ptr<BaseSam2TrackModel> &sam2_model,
                           const std::string                         &test_image_path,
                           const std::vector<std::pair<int, int>>    &points,
                           const std::string                         &test_result_path)
{
  auto    image = cv::imread(test_image_path);
  cv::Mat masks;
  CHECK(sam2_model->GenerateMask(image, points, std::vector<int>{1}, masks, false));

  if (!test_result_path.empty())
  {
    ImageDrawHelper helper(std::make_shared<cv::Mat>(image.clone()));
    helper.addRedMaskToForeground(masks);
    cv::imwrite(test_result_path, *helper.getImage());
  }
}

void sam2_test_register_correctness(const std::shared_ptr<BaseSam2TrackModel> &sam2_model,
                                    const std::string                         &test_image_path,
                                    const std::vector<std::pair<int, int>>    &points,
                                    const std::string                         &test_result_path)
{
  auto image = cv::imread(test_image_path);

  CHECK(sam2_model->SetImage(image));

  std::unordered_map<size_t, cv::Mat> masks;
  CHECK(sam2_model->Register(points, std::vector<int>{1}, masks));
  CHECK(masks.size() == 1);

  if (!test_result_path.empty())
  {
    ImageDrawHelper helper(std::make_shared<cv::Mat>(image.clone()));
    helper.addRedMaskToForeground(masks.begin()->second);
    cv::imwrite(test_result_path, *helper.getImage());
  }
}

void sam2_test_track_correctness(const std::shared_ptr<BaseSam2TrackModel> &sam2_model,
                                 const std::string                         &test_images_dir_path,
                                 const std::vector<std::pair<int, int>>    &points,
                                 const std::string                         &test_result_dir_path)
{
  auto rgb_paths = get_files_in_directory(test_images_dir_path);
  std::sort(rgb_paths.begin(), rgb_paths.end());

  for (size_t i = 0; i < rgb_paths.size(); ++i)
  {
    cv::Mat image = cv::imread(rgb_paths[i].string());
    CHECK(!image.empty());

    CHECK(sam2_model->SetImage(image));

    std::unordered_map<size_t, cv::Mat> masks;
    if (i == 0)
    {
      CHECK(sam2_model->Register({points[0]}, std::vector<int>{1}, masks));
      CHECK(sam2_model->Register({points[1]}, std::vector<int>{1}, masks));
    } else
    {
      sam2_model->Track(masks);
      CHECK(masks.size() == 2);
    }

    if (!test_result_dir_path.empty())
    {
      if (!std::filesystem::exists(test_result_dir_path))
      {
        std::filesystem::create_directory(test_result_dir_path);
      }
      ImageDrawHelper helper(std::make_shared<cv::Mat>(image.clone()));
      for (const auto &p_id_mask : masks)
      {
        helper.addRedMaskToForeground(p_id_mask.second);
      }
      cv::imwrite(test_result_dir_path / rgb_paths[i].filename(), *helper.getImage());
    }
  }
}

void sam2_test_track_speed(const std::shared_ptr<BaseSam2TrackModel> &sam2_model,
                           const std::vector<std::pair<int, int>>    &points,
                           const std::string                         &test_images_dir_path)
{
  auto rgb_paths = get_files_in_directory(test_images_dir_path);
  std::sort(rgb_paths.begin(), rgb_paths.end());

  FPSCounter fps_counter;
  fps_counter.Start();
  for (size_t i = 0; i < rgb_paths.size(); ++i)
  {
    cv::Mat image = cv::imread(rgb_paths[i].string());
    CHECK(!image.empty());

    CHECK(sam2_model->SetImage(image));

    std::unordered_map<size_t, cv::Mat> masks;
    if (i == 0)
    {
      CHECK(sam2_model->Register({points[0]}, std::vector<int>{1}, masks));
    } else
    {
      sam2_model->Track(masks);
      CHECK(masks.size() == 1);
    }

    fps_counter.Count(1);
    if (i % 50 == 0)
      LOG(WARNING) << "Average qps : " << fps_counter.GetFPS();
  }
}

#define GEN_TEST_CASES(Tag, FixtureClass)                                                    \
  TEST_F(FixtureClass, test_sam2_##Tag##_correctness)                                        \
  {                                                                                          \
    sam2_test_correctness(sam2_model_, test_image_path_, one_shot_image_points_,             \
                          test_sam2_result_path_);                                           \
  }                                                                                          \
  TEST_F(FixtureClass, test_sam2_##Tag##_register_correctness)                               \
  {                                                                                          \
    sam2_test_register_correctness(sam2_model_, test_image_path_, one_shot_image_points_,    \
                                   test_sam2_register_result_path_);                         \
  }                                                                                          \
  TEST_F(FixtureClass, test_sam2_##Tag##_track_correctness)                                  \
  {                                                                                          \
    sam2_test_track_correctness(sam2_model_, test_images_dir_path_, track_set_image_points_, \
                                test_sam2_track_result_dir_path_);                           \
  }                                                                                          \
  TEST_F(FixtureClass, test_sam2_##Tag##_track_speed)                                        \
  {                                                                                          \
    sam2_test_track_speed(sam2_model_, track_set_image_points_, test_images_dir_path_);      \
  }

class BaseSAM2TrackFixture : public testing::Test {
protected:
  std::shared_ptr<BaseSam2TrackModel> sam2_model_;

  std::vector<std::pair<int, int>> one_shot_image_points_;
  std::string                      test_image_path_;
  std::string                      test_sam2_result_path_;
  std::string                      test_sam2_register_result_path_;

  std::vector<std::pair<int, int>> track_set_image_points_;
  std::string                      test_images_dir_path_;
  std::string                      test_sam2_track_result_dir_path_;
};

#ifdef ENABLE_TENSORRT

#include "trt_core/trt_core.h"

class Sam2_TensorRT_Fixture : public BaseSAM2TrackFixture {
public:
  void SetUp() override
  {
    auto encoder_engine = CreateTrtInferCore("/workspace/models/image_encoder.engine");
    auto decoder_engine = CreateTrtInferCore("/workspace/models/image_decoder.engine");
    auto memory_attention_engine =
        CreateTrtInferCore("/workspace/models/memory_attention_opt.engine",
                           {{"current_vision_feat", {1, 256, 64, 64}},
                            {"current_vision_pos_embed", {4096, 1, 256}},
                            {"memory_0", {16, 256}},
                            {"memory_1", {7, 64, 64, 64}},
                            {"memory_pos_embed", {28736, 1, 64}}},
                           {{"image_embed", {1, 256, 64, 64}}});
    auto memory_encoder_engine = CreateTrtInferCore("/workspace/models/memory_encoder.engine");
    auto preprocess_block =
        CreateCpuDetPreProcess({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, true, true);

    sam2_model_ = sam::CreateSam2Model(encoder_engine, decoder_engine, memory_attention_engine,
                                       memory_encoder_engine, preprocess_block);

    one_shot_image_points_           = {{225, 370}};
    test_image_path_                 = "/workspace/test_data/persons.jpg";
    test_sam2_result_path_           = "/workspace/test_data/test_sam2_trt_result.png";
    test_sam2_register_result_path_  = "/workspace/test_data/test_sam2_trt_register_result.png";
    track_set_image_points_          = {{420, 220}, {120, 220}};
    test_images_dir_path_            = "/workspace/test_data/golf/";
    test_sam2_track_result_dir_path_ = "/workspace/test_data/test_sam2_trt_track_result/";
  }
};

GEN_TEST_CASES(tensorrt, Sam2_TensorRT_Fixture);

#endif

#ifdef ENABLE_ORT

#include "ort_core/ort_core.h"

class Sam2_OnnxRuntime_Fixture : public BaseSAM2TrackFixture {
public:
  void SetUp() override
  {
    auto encoder_engine          = CreateOrtInferCore("/workspace/models/image_encoder.onnx");
    auto decoder_engine          = CreateOrtInferCore("/workspace/models/image_decoder.onnx",
                                                      {
                                                        {"point_coords", {1, 8, 2}},
                                                        {"point_labels", {1, 8}},
                                                        {"image_embed", {1, 256, 64, 64}},
                                                        {"high_res_feats_0", {1, 32, 256, 256}},
                                                        {"high_res_feats_1", {1, 64, 128, 128}}
                                                      },
                                                    {
                                                      {"obj_ptr", {1, 256}},
                                                      {"mask_for_mem", {1, 1, 1024, 1024}},
                                                      {"pred_mask", {1, 1, 1024, 1024}}
                                                    });
    auto memory_attention_engine = CreateOrtInferCore("/workspace/models/memory_attention_opt.onnx",
                                                      {{"current_vision_feat", {1, 256, 64, 64}},
                                                       {"current_vision_pos_embed", {4096, 1, 256}},
                                                       {"memory_0", {16, 256}},
                                                       {"memory_1", {7, 64, 64, 64}},
                                                       {"memory_pos_embed", {28736, 1, 64}}},
                                                      {{"image_embed", {1, 256, 64, 64}}});
    auto memory_encoder_engine   = CreateOrtInferCore("/workspace/models/memory_encoder.onnx");
    auto preprocess_block =
        CreateCpuDetPreProcess({123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}, true, true);

    sam2_model_ = sam::CreateSam2Model(encoder_engine, decoder_engine, memory_attention_engine,
                                       memory_encoder_engine, preprocess_block);

    one_shot_image_points_           = {{225, 370}};
    test_image_path_                 = "/workspace/test_data/persons.jpg";
    test_sam2_result_path_           = "/workspace/test_data/test_sam2_ort_result.png";
    test_sam2_register_result_path_  = "/workspace/test_data/test_sam2_ort_register_result.png";
    track_set_image_points_          = {{420, 220}, {120, 220}};
    test_images_dir_path_            = "/workspace/test_data/golf/";
    test_sam2_track_result_dir_path_ = "/workspace/test_data/test_sam2_ort_track_result/";
  }
};

GEN_TEST_CASES(onnxruntime, Sam2_OnnxRuntime_Fixture);

#endif