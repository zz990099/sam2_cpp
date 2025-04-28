#pragma once

#include "sam2/sam2.hpp"

namespace sam {

class Sam2 : public BaseSam2TrackModel {
public:
  Sam2(std::shared_ptr<inference_core::BaseInferCore>      image_encoder_core,
       std::shared_ptr<inference_core::BaseInferCore>      image_decoder_core,
       std::shared_ptr<inference_core::BaseInferCore>      memory_attention_core,
       std::shared_ptr<inference_core::BaseInferCore>      memory_encoder_core,
       std::shared_ptr<detection_2d::IDetectionPreProcess> image_preprocess_block,
       const std::vector<std::string>                     &image_encoder_blob_names,
       const std::vector<std::string>                     &image_decoder_blob_names,
       const std::vector<std::string>                     &memory_attention_blob_names,
       const std::vector<std::string>                     &memory_encoder_blob_names);

  ~Sam2() = default;

private:
  // SAM functionality
  bool ImagePreProcess(ParsingType pipeline_unit) override;

  bool PromptBoxPreProcess(ParsingType pipeline_unit) override;

  bool PromptPointPreProcess(ParsingType pipeline_unit) override;

  bool MaskPostProcess(ParsingType pipeline_unit) override;

  // SAM2 Track functionality
  bool Register(const cv::Mat                          &image,
                const std::vector<std::pair<int, int>> &points,
                const std::vector<int>                 &labels,
                cv::Mat                                &result,
                bool                                    isRGB = false) override;

  bool Register(const cv::Mat             &image,
                const std::vector<BBox2D> &boxes,
                cv::Mat                   &result,
                bool                       isRGB = false) override;

  bool Track(const cv::Mat &image, cv::Mat &result, bool isRGB = false) override;

private:
  // Track Records
  std::map<size_t, std::vector<float>> obj_ptr_pool_;         // 1, 256
  std::map<size_t, std::vector<float>> maskmem_feats_pool_;   // 1, 64, 64, 64
  std::map<size_t, std::vector<float>> maskmem_pos_enc_pool_; // 4096, 1, 64
  std::vector<float> time_embeding_;                          // 7, 1, 64
  size_t frame_idx_;

private:
  static const std::string                            model_name_;
  std::shared_ptr<inference_core::BaseInferCore>      image_encoder_core_;
  std::shared_ptr<inference_core::BaseInferCore>      image_decoder_core_;
  std::shared_ptr<inference_core::BaseInferCore>      memory_attention_core_;
  std::shared_ptr<inference_core::BaseInferCore>      memory_encoder_core_;
  std::shared_ptr<detection_2d::IDetectionPreProcess> image_preprocess_block_;

  const std::vector<std::string> image_encoder_blob_names_;
  const std::vector<std::string> image_decoder_blob_names_;
  const std::vector<std::string> memory_attention_blob_names_;
  const std::vector<std::string> memory_encoder_blob_names_;

private:
  // defualt params, no access provided to user
  const int IMAGE_INPUT_HEIGHT   = 1024;
  const int IMAGE_INPUT_WIDTH    = 1024;
  const int IMAGE_FEATURE_HEIGHT = 256;
  const int IMAGE_FEATURE_WIDTH  = 256;
  const int IMAGE_FEATURES_LEN   = 32;
  const int MASK_LOW_RES_HEIGHT  = 256;
  const int MASK_LOW_RES_WIDTH   = 256;
};

} // namespace sam