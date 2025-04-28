#include "sam2_def.hpp"

#include "deploy_core/wrapper.h"

namespace sam {

bool Sam2::Register(const cv::Mat                          &image,
                    const std::vector<std::pair<int, int>> &points,
                    const std::vector<int>                 &labels,
                    cv::Mat                                &result,
                    bool                                    isRGB)
{
  CHECK_STATE(!image.empty(), "[Sam2] Register got invalid image input!");
  CHECK_STATE(points.size() >= 1 && labels.size() == points.size(),
              "[Sam2] Register got invalid points & labels input!");

  // 1. get blobs buffer
  auto image_encoder_blobs_buffer    = image_encoder_core_->GetBuffer(true);
  auto image_decoder_blobs_buffer    = image_decoder_core_->GetBuffer(true);
  auto memory_attention_blobs_buffer = memory_attention_core_->GetBuffer(true);
  auto memory_encoder_blobs_buffer   = memory_encoder_core_->GetBuffer(true);

  // 2. image preprocess
  auto        image_wrapper = std::make_shared<PipelineCvImageWrapper>(image, isRGB);
  const float scale = image_preprocess_block_->Preprocess(image_wrapper, image_encoder_blobs_buffer,
                                                          image_encoder_blob_names_[0],
                                                          IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH);

  // 3. image encoder
  image_encoder_blobs_buffer->SetBlobBuffer(image_encoder_blob_names_[1], DataLocation::DEVICE);
  image_encoder_blobs_buffer->SetBlobBuffer(image_encoder_blob_names_[2], DataLocation::DEVICE);
  image_encoder_blobs_buffer->SetBlobBuffer(image_encoder_blob_names_[3], DataLocation::DEVICE);
  image_encoder_blobs_buffer->SetBlobBuffer(image_encoder_blob_names_[4], DataLocation::DEVICE);
  image_encoder_blobs_buffer->SetBlobBuffer(image_encoder_blob_names_[5], DataLocation::DEVICE);
  CHECK_STATE(image_encoder_core_->SyncInfer(image_encoder_blobs_buffer),
              "[Sam2] Register image_encoder sync infer failed!!!");

  // 4. prompt preprocess
  // 4.1 non-copy buffer reuse, 'high_res_feat0', 'high_res_feat1', 'vision_feats'(->image_embed)
  image_decoder_blobs_buffer->SetBlobBuffer(
      image_decoder_blob_names_[1],
      image_encoder_blobs_buffer->GetOuterBlobBuffer(image_encoder_blob_names_[1]).first,
      DataLocation::DEVICE);
  image_decoder_blobs_buffer->SetBlobBuffer(
      image_decoder_blob_names_[2],
      image_encoder_blobs_buffer->GetOuterBlobBuffer(image_encoder_blob_names_[2]).first,
      DataLocation::DEVICE);
  image_decoder_blobs_buffer->SetBlobBuffer(
      image_decoder_blob_names_[0],
      image_encoder_blobs_buffer->GetOuterBlobBuffer(image_encoder_blob_names_[3]).first,
      DataLocation::DEVICE);

  // 4.2 point prompt
  float *points_ptr = reinterpret_cast<float *>(
      image_decoder_blobs_buffer->GetOuterBlobBuffer(image_decoder_blob_names_[3]).first);
  float *labels_ptr = reinterpret_cast<float *>(
      image_decoder_blobs_buffer->GetOuterBlobBuffer(image_decoder_blob_names_[4]).first);
  const int64_t dynamic_point_number = points.size();
  for (int i = 0; i < dynamic_point_number; ++i)
  {
    const auto &point     = points[i];
    const auto &lab       = labels[i];
    points_ptr[i * 2 + 0] = static_cast<float>(point.first * scale);
    points_ptr[i * 2 + 1] = static_cast<float>(point.second * scale);
    labels_ptr[i]         = static_cast<float>(lab);
  }
  std::vector<int64_t> coords_dynamic_shape{1, dynamic_point_number, 2};
  image_decoder_blobs_buffer->SetBlobShape(image_decoder_blob_names_[3], coords_dynamic_shape);
  std::vector<int64_t> labels_dynamic_shape{1, dynamic_point_number};
  image_decoder_blobs_buffer->SetBlobShape(image_decoder_blob_names_[4], labels_dynamic_shape);

  // 5. image_decoder
  image_decoder_blobs_buffer->SetBlobBuffer(image_decoder_blob_names_[5], DataLocation::HOST);
  image_decoder_blobs_buffer->SetBlobBuffer(image_decoder_blob_names_[6], DataLocation::DEVICE);
  image_decoder_blobs_buffer->SetBlobBuffer(image_decoder_blob_names_[7], DataLocation::HOST);
  MESSURE_DURATION_AND_CHECK_STATE(image_decoder_core_->SyncInfer(image_decoder_blobs_buffer),
                                   "[Sam2] Register image_decoder sync infer failed!!!");

  // 6. Get pred masks
  void *decoder_output_masks_ptr =
      image_decoder_blobs_buffer->GetOuterBlobBuffer(image_decoder_blob_names_[7]).first;
  cv::Mat masks_output(IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, CV_32FC1, decoder_output_masks_ptr);
  // 6.1 crop valid block
  masks_output = masks_output(cv::Range(0, image.rows * scale), cv::Range(0, image.cols * scale));
  // 6.2 resize to original size
  cv::resize(masks_output, masks_output, {image.cols, image.rows}, 0, 0, cv::INTER_LINEAR);
  // 6.3 convert to binary mask
  cv::threshold(masks_output, masks_output, 0, 255, cv::THRESH_BINARY);
  // 6.4 convert to CV_8U
  masks_output.convertTo(masks_output, CV_8U);
  // 6.5 output
  result = masks_output;

  // 7. memory_encoder
  memory_encoder_blobs_buffer->SetBlobBuffer(
      memory_encoder_blob_names_[0],
      image_decoder_blobs_buffer->GetOuterBlobBuffer(image_decoder_blob_names_[6]).first,
      DataLocation::DEVICE);
  memory_encoder_blobs_buffer->SetBlobBuffer(
      memory_encoder_blob_names_[1],
      image_encoder_blobs_buffer->GetOuterBlobBuffer(image_encoder_blob_names_[5]).first,
      DataLocation::DEVICE);
  MESSURE_DURATION_AND_CHECK_STATE(memory_encoder_core_->SyncInfer(memory_encoder_blobs_buffer),
                                   "[Sam2] Register memory_encoder sync infer failed!!!");

  // 8. record!
  std::vector<float> obj_ptr(256);
  std::vector<float> maskmem_feat(64 * 64 * 64);
  std::vector<float> maskmem_pos_enc(4096 * 64);
  std::vector<float> time_embeding(7 * 64);
  // 8.1 copy buffer
  memcpy(obj_ptr.data(),
         image_decoder_blobs_buffer->GetOuterBlobBuffer(image_decoder_blob_names_[5]).first,
         obj_ptr.size() * sizeof(float));
  memcpy(maskmem_feat.data(),
         memory_encoder_blobs_buffer->GetOuterBlobBuffer(memory_encoder_blob_names_[2]).first,
         maskmem_feat.size() * sizeof(float));
  memcpy(maskmem_pos_enc.data(),
         memory_encoder_blobs_buffer->GetOuterBlobBuffer(memory_encoder_blob_names_[3]).first,
         maskmem_pos_enc.size() * sizeof(float));
  memcpy(time_embeding.data(),
         memory_encoder_blobs_buffer->GetOuterBlobBuffer(memory_encoder_blob_names_[4]).first,
         time_embeding.size() * sizeof(float));
  // 8.2 record in global member
  obj_ptr_pool_.clear();
  maskmem_feats_pool_.clear();
  maskmem_pos_enc_pool_.clear();
  frame_idx_ = 0;
  obj_ptr_pool_.emplace(frame_idx_, std::move(obj_ptr));
  maskmem_feats_pool_.emplace(frame_idx_, std::move(maskmem_feat));
  maskmem_pos_enc_pool_.emplace(frame_idx_, std::move(maskmem_pos_enc));
  time_embeding_ = std::move(time_embeding);
  ++frame_idx_;

  return true;
}

bool Sam2::Register(const cv::Mat             &image,
                    const std::vector<BBox2D> &boxes,
                    cv::Mat                   &result,
                    bool                       isRGB)
{
  return true;
}

bool Sam2::Track(const cv::Mat &image, cv::Mat &result, bool isRGB)
{
  CHECK_STATE(!image.empty(), "[Sam2] Track got invalid image input!");

  // 1. get blobs buffer
  auto image_encoder_blobs_buffer    = image_encoder_core_->GetBuffer(true);
  auto image_decoder_blobs_buffer    = image_decoder_core_->GetBuffer(true);
  auto memory_attention_blobs_buffer = memory_attention_core_->GetBuffer(true);
  auto memory_encoder_blobs_buffer   = memory_encoder_core_->GetBuffer(true);

  // 2. image preprocess
  auto        image_wrapper = std::make_shared<PipelineCvImageWrapper>(image, isRGB);
  const float scale = image_preprocess_block_->Preprocess(image_wrapper, image_encoder_blobs_buffer,
                                                          image_encoder_blob_names_[0],
                                                          IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH);

  // 3. image encoder
  image_encoder_blobs_buffer->SetBlobBuffer(image_encoder_blob_names_[1], DataLocation::DEVICE);
  image_encoder_blobs_buffer->SetBlobBuffer(image_encoder_blob_names_[2], DataLocation::DEVICE);
  image_encoder_blobs_buffer->SetBlobBuffer(image_encoder_blob_names_[3], DataLocation::DEVICE);
  image_encoder_blobs_buffer->SetBlobBuffer(image_encoder_blob_names_[4], DataLocation::DEVICE);
  image_encoder_blobs_buffer->SetBlobBuffer(image_encoder_blob_names_[5], DataLocation::DEVICE);
  MESSURE_DURATION_AND_CHECK_STATE(image_encoder_core_->SyncInfer(image_encoder_blobs_buffer),
                                   "[Sam2] Track image_encoder sync infer failed!!!");

  // 4. memory_attention
  // 4.1 vision_feats & vision_pos_embed
  memory_attention_blobs_buffer->SetBlobBuffer(
      memory_attention_blob_names_[0],
      image_encoder_blobs_buffer->GetOuterBlobBuffer(image_encoder_blob_names_[3]).first,
      DataLocation::DEVICE);
  memory_attention_blobs_buffer->SetBlobBuffer(
      memory_attention_blob_names_[1],
      image_encoder_blobs_buffer->GetOuterBlobBuffer(image_encoder_blob_names_[4]).first,
      DataLocation::DEVICE);
  // 4.2 memory_0
  float *memory_0_ptr = reinterpret_cast<float *>(
      memory_attention_blobs_buffer->GetOuterBlobBuffer(memory_attention_blob_names_[2]).first);
  size_t memory_0_offset = 0;
  for (const auto &p_id_obj_ptr : obj_ptr_pool_)
  {
    const auto &obj_ptr = p_id_obj_ptr.second;
    memcpy(memory_0_ptr + memory_0_offset, obj_ptr.data(), obj_ptr.size() * sizeof(float));
    memory_0_offset += obj_ptr.size();
  }
  const int64_t        obj_ptr_num = obj_ptr_pool_.size();
  std::vector<int64_t> memory_0_shape{obj_ptr_num, 256};
  memory_attention_blobs_buffer->SetBlobShape(memory_attention_blob_names_[2], memory_0_shape);
  // 4.3 memory_1
  float *memory_1_ptr = reinterpret_cast<float *>(
      memory_attention_blobs_buffer->GetOuterBlobBuffer(memory_attention_blob_names_[3]).first);
  size_t memory_1_offset = 0;
  for (const auto &p_id_maskmem_feats : maskmem_feats_pool_)
  {
    const auto &maskmem_feats = p_id_maskmem_feats.second;
    memcpy(memory_1_ptr + memory_1_offset, maskmem_feats.data(),
           maskmem_feats.size() * sizeof(float));
    memory_1_offset += maskmem_feats.size();
  }
  const int64_t        memory_1_num = maskmem_feats_pool_.size();
  std::vector<int64_t> memory_1_shape{memory_1_num, 64, 64, 64};
  memory_attention_blobs_buffer->SetBlobShape(memory_attention_blob_names_[3], memory_1_shape);
  // 4.4 memory_pos_embed
  float *memory_pos_embed_ptr = reinterpret_cast<float *>(
      memory_attention_blobs_buffer->GetOuterBlobBuffer(memory_attention_blob_names_[4]).first);
  size_t memory_pos_embed_offset = 0;
  size_t time_embeding_offset    = 0;
  for (const auto &p_id_memory_pos_embed : maskmem_pos_enc_pool_)
  {
    const auto &memory_pos_embed           = p_id_memory_pos_embed.second;
    float      *local_memory_pos_embed_ptr = memory_pos_embed_ptr + memory_pos_embed_offset;

    for (size_t i = 0; i < 4096; ++i)
    {
      size_t idx = i * 64;
      for (size_t j = 0; j < 64; ++j)
      {
        local_memory_pos_embed_ptr[idx + j] =
            memory_pos_embed[idx + j] + time_embeding_[time_embeding_offset + j];
      }
    }

    memory_pos_embed_offset += memory_pos_embed.size();
    time_embeding_offset += 64;
  }
  memset(memory_pos_embed_ptr + memory_pos_embed_offset, 0,
         obj_ptr_pool_.size() * 4 * sizeof(float));
  const int64_t memory_pos_embed_num =
      maskmem_pos_enc_pool_.size() * 4096 + obj_ptr_pool_.size() * 4;
  std::vector<int64_t> memory_pos_embed_shape{memory_pos_embed_num, 1, 64};
  memory_attention_blobs_buffer->SetBlobShape(memory_attention_blob_names_[4],
                                              memory_pos_embed_shape);
  // 4.5 memory attention inference
  memory_attention_blobs_buffer->SetBlobBuffer(memory_attention_blob_names_[5],
                                               DataLocation::DEVICE);
  MESSURE_DURATION_AND_CHECK_STATE(memory_attention_core_->SyncInfer(memory_attention_blobs_buffer),
                                   "[Sam2] Register memory_attention sync infer failed!!!");

  // 5. image decoder
  // 5.1 设置缓存指针
  image_decoder_blobs_buffer->SetBlobBuffer(
      image_decoder_blob_names_[1],
      image_encoder_blobs_buffer->GetOuterBlobBuffer(image_encoder_blob_names_[1]).first,
      DataLocation::DEVICE);
  image_decoder_blobs_buffer->SetBlobBuffer(
      image_decoder_blob_names_[2],
      image_encoder_blobs_buffer->GetOuterBlobBuffer(image_encoder_blob_names_[2]).first,
      DataLocation::DEVICE);
  image_decoder_blobs_buffer->SetBlobBuffer(
      image_decoder_blob_names_[0],
      memory_attention_blobs_buffer->GetOuterBlobBuffer(memory_attention_blob_names_[5]).first,
      DataLocation::DEVICE);
  // 5.2 Set prompt
  float *points_ptr = reinterpret_cast<float *>(
      image_decoder_blobs_buffer->GetOuterBlobBuffer(image_decoder_blob_names_[3]).first);
  // 5.3 point labels
  float *labels_ptr = reinterpret_cast<float *>(
      image_decoder_blobs_buffer->GetOuterBlobBuffer(image_decoder_blob_names_[4]).first);
  points_ptr[0] = 0.f;
  points_ptr[1] = 0.f;
  labels_ptr[0] = -1.f;
  // 5.4 Set dynamic shape
  std::vector<int64_t> coords_dynamic_shape{1, 1, 2};
  image_decoder_blobs_buffer->SetBlobShape(image_decoder_blob_names_[3], coords_dynamic_shape);
  std::vector<int64_t> labels_dynamic_shape{1, 1};
  image_decoder_blobs_buffer->SetBlobShape(image_decoder_blob_names_[4], labels_dynamic_shape);
  // let unused buffer kept on device side
  image_decoder_blobs_buffer->SetBlobBuffer(image_decoder_blob_names_[5], DataLocation::HOST);
  image_decoder_blobs_buffer->SetBlobBuffer(image_decoder_blob_names_[6], DataLocation::DEVICE);
  image_decoder_blobs_buffer->SetBlobBuffer(image_decoder_blob_names_[7], DataLocation::HOST);
  // 5.5 image decoder inference
  MESSURE_DURATION_AND_CHECK_STATE(image_decoder_core_->SyncInfer(image_decoder_blobs_buffer),
                                   "[Sam2] Register image_decoder sync infer failed!!!");
  // 5.6 Get pred masks
  void *decoder_output_masks_ptr =
      image_decoder_blobs_buffer->GetOuterBlobBuffer(image_decoder_blob_names_[7]).first;
  cv::Mat masks_output(IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, CV_32FC1, decoder_output_masks_ptr);
  // 5.6.1 crop valid block
  masks_output = masks_output(cv::Range(0, image.rows * scale), cv::Range(0, image.cols * scale));
  // 5.6.2 resize to original size
  cv::resize(masks_output, masks_output, {image.cols, image.rows}, 0, 0, cv::INTER_LINEAR);
  // 5.6.3 convert to binary mask
  cv::threshold(masks_output, masks_output, 0, 255, cv::THRESH_BINARY);
  // 5.6.4 convert to CV_8U
  masks_output.convertTo(masks_output, CV_8U);
  // 5.6.5 output
  result = masks_output;

  // 6. memory encoder
  memory_encoder_blobs_buffer->SetBlobBuffer(
      memory_encoder_blob_names_[0],
      image_decoder_blobs_buffer->GetOuterBlobBuffer(image_decoder_blob_names_[6]).first,
      DataLocation::DEVICE);
  memory_encoder_blobs_buffer->SetBlobBuffer(
      memory_encoder_blob_names_[1],
      image_encoder_blobs_buffer->GetOuterBlobBuffer(image_encoder_blob_names_[5]).first,
      DataLocation::DEVICE);
  MESSURE_DURATION_AND_CHECK_STATE(memory_encoder_core_->SyncInfer(memory_encoder_blobs_buffer),
                                   "[Sam2] Register memory_encoder sync infer failed!!!");

  // 7. record!
  std::vector<float> obj_ptr(256);
  std::vector<float> maskmem_feat(64 * 64 * 64);
  std::vector<float> maskmem_pos_enc(4096 * 64);
  std::vector<float> time_embeding(7 * 64);
  // 8.1 copy buffer
  memcpy(obj_ptr.data(),
         image_decoder_blobs_buffer->GetOuterBlobBuffer(image_decoder_blob_names_[5]).first,
         obj_ptr.size() * sizeof(float));
  memcpy(maskmem_feat.data(),
         memory_encoder_blobs_buffer->GetOuterBlobBuffer(memory_encoder_blob_names_[2]).first,
         maskmem_feat.size() * sizeof(float));
  memcpy(maskmem_pos_enc.data(),
         memory_encoder_blobs_buffer->GetOuterBlobBuffer(memory_encoder_blob_names_[3]).first,
         maskmem_pos_enc.size() * sizeof(float));
  memcpy(time_embeding.data(),
         memory_encoder_blobs_buffer->GetOuterBlobBuffer(memory_encoder_blob_names_[4]).first,
         time_embeding.size() * sizeof(float));
  // 7.1 obj_ptr
  if (obj_ptr_pool_.size() >= 16)
  {
    auto second_itr = ++obj_ptr_pool_.begin();
    obj_ptr_pool_.erase(second_itr);
  }
  obj_ptr_pool_.emplace(frame_idx_, std::move(obj_ptr));
  // 7.2 maskmem_feats
  if (maskmem_feats_pool_.size() >= 7)
  {
    maskmem_feats_pool_.erase(maskmem_feats_pool_.begin());
  }
  maskmem_feats_pool_.emplace(frame_idx_, std::move(maskmem_feat));
  // 7.3 maskmem_pos_enc
  if (maskmem_pos_enc_pool_.size() >= 7)
  {
    maskmem_pos_enc_pool_.erase(maskmem_pos_enc_pool_.begin());
  }
  maskmem_pos_enc_pool_.emplace(frame_idx_, std::move(maskmem_pos_enc));
  // 7.4 time_embeding
  time_embeding_ = std::move(time_embeding);
  // 7.5 frame_idx
  ++frame_idx_;

  return true;
}

} // namespace sam