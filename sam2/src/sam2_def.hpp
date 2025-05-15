#pragma once

#include "sam2/sam2.hpp"

namespace sam {

enum Sam2Mode { Normal = 0, Register = 1, Track = 2 };

struct Sam2TrackPipelinePackage : public SamPipelinePackage {
  std::shared_ptr<inference_core::BlobsTensor> memory_attention_blobs_buffer;
  std::shared_ptr<inference_core::BlobsTensor> memory_encoder_blobs_buffer;
  Sam2Mode                                     current_mode;
  size_t                                       current_frame_idx;
  size_t                                       current_obj_id;
};

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

  ~Sam2()
  {
    current_package_.reset();
  }

  // SAM2 Track functionality
  bool SetImage(const cv::Mat &image, bool isRGB = false) override;

  bool Register(const std::vector<std::pair<int, int>> &points,
                const std::vector<int>                 &labels,
                std::unordered_map<size_t, cv::Mat>    &results) override;

  bool Register(const std::vector<BBox2D>           &boxes,
                std::unordered_map<size_t, cv::Mat> &results) override;

  bool Track(std::unordered_map<size_t, cv::Mat> &results) override;

private:
  // SAM functionality
  bool ImagePreProcess(ParsingType pipeline_unit) override;

  bool PromptBoxPreProcess(ParsingType pipeline_unit) override;

  bool PromptPointPreProcess(ParsingType pipeline_unit) override;

  bool MaskPostProcess(ParsingType pipeline_unit) override;

private:
  bool RunMemoryEncoderAndRecord(ParsingType pipeline_unit);

  bool MemoryAttentionPreProcess(ParsingType pipeline_unit);

  bool PromptPointPreProcessTrackMode(ParsingType pipeline_unit);

private:
  struct MemoryDict {
    // Track Records
    std::map<size_t, std::vector<float>> obj_ptr_pool_;         // 1, 256
    std::map<size_t, std::vector<float>> maskmem_feats_pool_;   // 1, 64, 64, 64
    std::map<size_t, std::vector<float>> maskmem_pos_enc_pool_; // 4096, 1, 64
    std::vector<float>                   time_embeding_;        // 7, 1, 64
  };
  std::unordered_map<size_t, MemoryDict>    memory_bank_;
  std::shared_ptr<Sam2TrackPipelinePackage> current_package_;

  size_t frame_idx_{0ul};
  size_t obj_id_{0ul};

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

  const size_t OBJ_PTR_DIM          = 256;
  const size_t OBJ_PTR_POOL_SIZE    = 16;
  const size_t MASK_MEM_HEIGHT      = 64;
  const size_t MASK_MEM_WIDTH       = 64;
  const size_t MASK_MEM_FEAT_DIM    = 64;
  const size_t MASK_MEM_POS_ENC_DIM = 64;
  const size_t MASK_MEM_POOL_SIZE   = 7;
  const size_t TIME_EMBEDING_DIM    = 64;
};

} // namespace sam