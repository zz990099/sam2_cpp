#pragma once

#include "deploy_core/base_sam.h"
#include "deploy_core/base_detection.h"

namespace sam {

class BaseSam2TrackModel : public BaseSamModel {
public:
  virtual ~BaseSam2TrackModel() = default;

  virtual bool SetImage(const cv::Mat &image, bool isRGB = false) = 0;

  virtual bool Register(const std::vector<std::pair<int, int>> &points,
                        const std::vector<int>                 &labels,
                        std::unordered_map<size_t, cv::Mat>    &results) = 0;

  virtual bool Register(const std::vector<BBox2D>           &boxes,
                        std::unordered_map<size_t, cv::Mat> &results) = 0;

  virtual bool Track(std::unordered_map<size_t, cv::Mat> &results) = 0;

protected:
  BaseSam2TrackModel(const std::string                             &model_name,
                     std::shared_ptr<inference_core::BaseInferCore> image_encoder_core,
                     std::shared_ptr<inference_core::BaseInferCore> mask_points_decoder_core,
                     std::shared_ptr<inference_core::BaseInferCore> mask_boxes_decoder_core)
      : BaseSamModel(
            model_name, image_encoder_core, mask_points_decoder_core, mask_boxes_decoder_core)
  {}
};

std::shared_ptr<BaseSam2TrackModel> CreateSam2Model(
    std::shared_ptr<inference_core::BaseInferCore>      image_encoder_core,
    std::shared_ptr<inference_core::BaseInferCore>      image_decoder_core,
    std::shared_ptr<inference_core::BaseInferCore>      memory_attention_core,
    std::shared_ptr<inference_core::BaseInferCore>      memory_encoder_core,
    std::shared_ptr<detection_2d::IDetectionPreProcess> image_preprocess_block,
    const std::vector<std::string> &image_encoder_blob_names = {"image", "high_res_feat0",
                                                                "high_res_feat1", "vision_feats",
                                                                "vision_pos_embed", "pix_feat"},
    const std::vector<std::string> &image_dec_blob_names     = {"image_embed", "high_res_feats_0",
                                                                "high_res_feats_1", "point_coords",
                                                                "point_labels", "obj_ptr",
                                                                "mask_for_mem", "pred_mask"},
    const std::vector<std::string> &memory_attention_blob_names = {"current_vision_feat",
                                                                   "current_vision_pos_embed",
                                                                   "memory_0", "memory_1",
                                                                   "memory_pos_embed",
                                                                   "image_embed"},
    const std::vector<std::string> &memory_encoder_blob_names   = {
        "mask_for_mem", "pix_feat", "maskmem_features", "maskmem_pos_enc", "temporal_code"});

} // namespace sam