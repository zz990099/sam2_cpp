#include "sam2_def.hpp"

#include "deploy_core/wrapper.h"

namespace sam {

bool Sam2::SetImage(const cv::Mat &image, bool isRGB)
{
  CHECK_STATE(!image.empty(), "[Sam2] SetImage got invalid image input!");

  auto image_encoder_blobs_buffer    = image_encoder_core_->GetBuffer(true);
  auto image_decoder_blobs_buffer    = image_decoder_core_->GetBuffer(true);
  auto memory_attention_blobs_buffer = memory_attention_core_->GetBuffer(true);
  auto memory_encoder_blobs_buffer   = memory_encoder_core_->GetBuffer(true);

  current_package_                   = std::make_shared<Sam2TrackPipelinePackage>();
  current_package_->input_image_data = std::make_shared<PipelineCvImageWrapper>(image, isRGB);
  current_package_->image_encoder_blobs_buffer    = image_encoder_blobs_buffer;
  current_package_->mask_decoder_blobs_buffer     = image_decoder_blobs_buffer;
  current_package_->memory_attention_blobs_buffer = memory_attention_blobs_buffer;
  current_package_->memory_encoder_blobs_buffer   = memory_encoder_blobs_buffer;
  current_package_->current_frame_idx             = ++frame_idx_;

  MESSURE_DURATION_AND_CHECK_STATE(ImagePreProcess(current_package_),
                                   "[Sam2-Register] Image-Preprocess execute failed!!!");

  MESSURE_DURATION_AND_CHECK_STATE(
      image_encoder_core_->SyncInfer(current_package_->GetInferBuffer()),
      "[Sam2-Register] Image-encoder sync infer execute failed!!!");

  return true;
}

bool Sam2::Register(const std::vector<std::pair<int, int>> &points,
                    const std::vector<int>                 &labels,
                    std::unordered_map<size_t, cv::Mat>    &results)
{
  CHECK_STATE(points.size() >= 1 && labels.size() == points.size(),
              "[Sam2] Register got invalid points & labels input!");
  current_package_->points       = points;
  current_package_->labels       = labels;
  current_package_->current_mode = Sam2Mode::Register;
  current_package_->current_obj_id = ++obj_id_;

  MESSURE_DURATION_AND_CHECK_STATE(PromptPointPreProcess(current_package_),
                                   "[Sam2-Register] Prompt-preprocess execute failed!!!");

  MESSURE_DURATION_AND_CHECK_STATE(
      image_decoder_core_->SyncInfer(current_package_->GetInferBuffer()),
      "[Sam2-Register] Register image_decoder sync infer failed!!!");

  MESSURE_DURATION_AND_CHECK_STATE(MaskPostProcess(current_package_),
                                   "[Sam2-Register] Mask-postprocess execute failed!!!");

  MESSURE_DURATION_AND_CHECK_STATE(_RunMemoryEncoderAndRecord(current_package_),
                                   "[Sam2-Register] Mask-postprocess execute failed!!!");

  results = {{obj_id_, current_package_->mask}};
  return true;
}

bool Sam2::_RunMemoryEncoderAndRecord(ParsingType pipeline_unit)
{
  auto p_package = std::dynamic_pointer_cast<Sam2TrackPipelinePackage>(pipeline_unit);
  CHECK_STATE(p_package != nullptr,
              "[Sam2Track] RunMemoryEncoderAndRecord the `package` instance \
                          is not a instance of `SamPipelinePackage`!");

  auto &image_encoder_blobs_buffer  = p_package->image_encoder_blobs_buffer;
  auto &image_decoder_blobs_buffer  = p_package->mask_decoder_blobs_buffer;
  auto &memory_encoder_blobs_buffer = p_package->memory_encoder_blobs_buffer;

  // 1.
  memory_encoder_blobs_buffer->SetBlobBuffer(
      memory_encoder_blob_names_[0],
      image_decoder_blobs_buffer->GetOuterBlobBuffer(image_decoder_blob_names_[6]).first,
      DataLocation::DEVICE);
  memory_encoder_blobs_buffer->SetBlobBuffer(
      memory_encoder_blob_names_[1],
      image_encoder_blobs_buffer->GetOuterBlobBuffer(image_encoder_blob_names_[5]).first,
      DataLocation::DEVICE);
  MESSURE_DURATION_AND_CHECK_STATE(
      memory_encoder_core_->SyncInfer(memory_encoder_blobs_buffer),
      "[Sam2Track] RunMemoryEncoderAndRecord memory_encoder sync infer failed!!!");

  // 2. record!
  std::vector<float> obj_ptr(OBJ_PTR_DIM);
  std::vector<float> maskmem_feat(MASK_MEM_HEIGHT * MASK_MEM_WIDTH * MASK_MEM_FEAT_DIM);
  std::vector<float> maskmem_pos_enc(MASK_MEM_HEIGHT * MASK_MEM_WIDTH * MASK_MEM_POS_ENC_DIM);
  std::vector<float> time_embeding(MASK_MEM_POOL_SIZE * TIME_EMBEDING_DIM);
  // 3. copy buffer
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
  // 4. record in global member
  MemoryDict &memory_dict = memory_bank_[p_package->current_obj_id];
  if (memory_dict.obj_ptr_pool_.size() >= 16)
  {
    auto second_itr = ++memory_dict.obj_ptr_pool_.begin();
    memory_dict.obj_ptr_pool_.erase(second_itr);
  }
  memory_dict.obj_ptr_pool_.emplace(frame_idx_, std::move(obj_ptr));
  // 5. maskmem_feats
  if (memory_dict.maskmem_feats_pool_.size() >= 7)
  {
    memory_dict.maskmem_feats_pool_.erase(memory_dict.maskmem_feats_pool_.begin());
  }
  memory_dict.maskmem_feats_pool_.emplace(frame_idx_, std::move(maskmem_feat));
  // 6. maskmem_pos_enc
  if (memory_dict.maskmem_pos_enc_pool_.size() >= 7)
  {
    memory_dict.maskmem_pos_enc_pool_.erase(memory_dict.maskmem_pos_enc_pool_.begin());
  }
  memory_dict.maskmem_pos_enc_pool_.emplace(frame_idx_, std::move(maskmem_pos_enc));
  // 7. time_embeding
  memory_dict.time_embeding_ = std::move(time_embeding);

  return true;
}

bool Sam2::_MemoryAttentionPreProcess(ParsingType pipeline_unit)
{
  auto p_package = std::dynamic_pointer_cast<Sam2TrackPipelinePackage>(pipeline_unit);
  CHECK_STATE(p_package != nullptr,
              "[Sam2Track] MemoryAttentionPreProcess the `package` instance \
                          is not a instance of `SamPipelinePackage`!");

  auto &memory_attention_blobs_buffer = p_package->memory_attention_blobs_buffer;
  auto &image_encoder_blobs_buffer    = p_package->image_encoder_blobs_buffer;

  // 1. vision_feats & vision_pos_embed
  memory_attention_blobs_buffer->SetBlobBuffer(
      memory_attention_blob_names_[0],
      image_encoder_blobs_buffer->GetOuterBlobBuffer(image_encoder_blob_names_[3]).first,
      DataLocation::DEVICE);
  memory_attention_blobs_buffer->SetBlobBuffer(
      memory_attention_blob_names_[1],
      image_encoder_blobs_buffer->GetOuterBlobBuffer(image_encoder_blob_names_[4]).first,
      DataLocation::DEVICE);

  // Get current obj id
  const auto &memory_dict = memory_bank_[p_package->current_obj_id];

  // 2. memory_0
  float *memory_0_ptr = reinterpret_cast<float *>(
      memory_attention_blobs_buffer->GetOuterBlobBuffer(memory_attention_blob_names_[2]).first);
  size_t memory_0_offset = 0;
  for (const auto &p_id_obj_ptr : memory_dict.obj_ptr_pool_)
  {
    const auto &obj_ptr = p_id_obj_ptr.second;
    memcpy(memory_0_ptr + memory_0_offset, obj_ptr.data(), obj_ptr.size() * sizeof(float));
    memory_0_offset += obj_ptr.size();
  }
  const int64_t        obj_ptr_num = memory_dict.obj_ptr_pool_.size();
  std::vector<int64_t> memory_0_shape{obj_ptr_num, 256};
  memory_attention_blobs_buffer->SetBlobShape(memory_attention_blob_names_[2], memory_0_shape);

  // 3. memory_1
  float *memory_1_ptr = reinterpret_cast<float *>(
      memory_attention_blobs_buffer->GetOuterBlobBuffer(memory_attention_blob_names_[3]).first);
  size_t memory_1_offset = 0;
  for (const auto &p_id_maskmem_feats : memory_dict.maskmem_feats_pool_)
  {
    const auto &maskmem_feats = p_id_maskmem_feats.second;
    memcpy(memory_1_ptr + memory_1_offset, maskmem_feats.data(),
           maskmem_feats.size() * sizeof(float));
    memory_1_offset += maskmem_feats.size();
  }
  const int64_t        memory_1_num = memory_dict.maskmem_feats_pool_.size();
  std::vector<int64_t> memory_1_shape{memory_1_num, static_cast<int64_t>(MASK_MEM_FEAT_DIM),
                                      static_cast<int64_t>(MASK_MEM_HEIGHT),
                                      static_cast<int64_t>(MASK_MEM_WIDTH)};
  memory_attention_blobs_buffer->SetBlobShape(memory_attention_blob_names_[3], memory_1_shape);

  // 4. memory_pos_embed
  float *memory_pos_embed_ptr = reinterpret_cast<float *>(
      memory_attention_blobs_buffer->GetOuterBlobBuffer(memory_attention_blob_names_[4]).first);
  const size_t mask_mem_pixel_count    = MASK_MEM_HEIGHT * MASK_MEM_WIDTH;
  size_t       memory_pos_embed_offset = 0;
  size_t       time_embeding_offset    = 0;
  for (const auto &p_id_memory_pos_embed : memory_dict.maskmem_pos_enc_pool_)
  {
    const auto &memory_pos_embed           = p_id_memory_pos_embed.second;
    float      *local_memory_pos_embed_ptr = memory_pos_embed_ptr + memory_pos_embed_offset;

    for (size_t i = 0; i < mask_mem_pixel_count; ++i)
    {
      size_t idx = i * MASK_MEM_POS_ENC_DIM;
      for (size_t j = 0; j < MASK_MEM_POS_ENC_DIM; ++j)
      {
        local_memory_pos_embed_ptr[idx + j] =
            memory_pos_embed[idx + j] + memory_dict.time_embeding_[time_embeding_offset + j];
      }
    }

    memory_pos_embed_offset += memory_pos_embed.size();
    time_embeding_offset += MASK_MEM_POS_ENC_DIM;
  }
  memset(memory_pos_embed_ptr + memory_pos_embed_offset, 0,
         memory_dict.obj_ptr_pool_.size() * 4 * sizeof(float));

  const int64_t memory_pos_embed_num =
      memory_dict.maskmem_pos_enc_pool_.size() * mask_mem_pixel_count +
      memory_dict.obj_ptr_pool_.size() * 4;
  std::vector<int64_t> memory_pos_embed_shape{memory_pos_embed_num, 1,
                                              static_cast<int64_t>(MASK_MEM_POS_ENC_DIM)};
  memory_attention_blobs_buffer->SetBlobShape(memory_attention_blob_names_[4],
                                              memory_pos_embed_shape);

  // 5. memory attention inference
  memory_attention_blobs_buffer->SetBlobBuffer(memory_attention_blob_names_[5],
                                               DataLocation::DEVICE);
  p_package->infer_buffer = memory_attention_blobs_buffer;

  return true;
}

bool Sam2::Register(const std::vector<BBox2D> &boxes, std::unordered_map<size_t, cv::Mat> &results)
{
  return true;
}

bool Sam2::Track(std::unordered_map<size_t, cv::Mat> &results)
{
  current_package_->current_mode = Sam2Mode::Track;
  results.clear();
  for (const auto &p_id_dict : memory_bank_)
  {
    const auto &obj_id      = p_id_dict.first;
    const auto &memory_dict = p_id_dict.second;

    current_package_->current_obj_id = obj_id;
    MESSURE_DURATION_AND_CHECK_STATE(_MemoryAttentionPreProcess(current_package_),
                                     "[Sam2-Track] MemoryAttentionPreProcess execute failed!!!");

    MESSURE_DURATION_AND_CHECK_STATE(
        memory_attention_core_->SyncInfer(current_package_->GetInferBuffer()),
        "[Sam2-Track] memory_attention sync infer failed!!!");

    current_package_->points = {{0, 0}};
    current_package_->labels = {-1};
    MESSURE_DURATION_AND_CHECK_STATE(PromptPointPreProcess(current_package_),
                                     "[Sam2-Track] Prompt-preprocess execute failed!!!");

    MESSURE_DURATION_AND_CHECK_STATE(
        image_decoder_core_->SyncInfer(current_package_->GetInferBuffer()),
        "[Sam2-Track] image_decoder sync infer failed!!!");

    MESSURE_DURATION_AND_CHECK_STATE(MaskPostProcess(current_package_),
                                     "[Sam2-Track] Mask-postprocess execute failed!!!");

    MESSURE_DURATION_AND_CHECK_STATE(_RunMemoryEncoderAndRecord(current_package_),
                                     "[Sam2-Track] Mask-postprocess execute failed!!!");

    results.emplace(obj_id, current_package_->mask);
  }

  return true;
}

} // namespace sam
