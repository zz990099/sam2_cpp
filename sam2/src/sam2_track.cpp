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
  current_package_->points         = points;
  current_package_->labels         = labels;
  current_package_->current_mode   = Sam2Mode::Register;
  current_package_->current_obj_id = ++obj_id_;

  MESSURE_DURATION_AND_CHECK_STATE(PromptPointPreProcessTrackMode(current_package_),
                                   "[Sam2-Register] Prompt-preprocess execute failed!!!");

  MESSURE_DURATION_AND_CHECK_STATE(
      image_decoder_core_->SyncInfer(current_package_->GetInferBuffer()),
      "[Sam2-Register] Register image_decoder sync infer failed!!!");

  MESSURE_DURATION_AND_CHECK_STATE(MaskPostProcess(current_package_),
                                   "[Sam2-Register] Mask-postprocess execute failed!!!");

  MESSURE_DURATION_AND_CHECK_STATE(RunMemoryEncoderAndRecord(current_package_),
                                   "[Sam2-Register] Mask-postprocess execute failed!!!");

  results = {{obj_id_, current_package_->mask}};
  return true;
}

bool Sam2::PromptPointPreProcessTrackMode(ParsingType package)
{
  auto p_package = std::dynamic_pointer_cast<Sam2TrackPipelinePackage>(package);
  CHECK_STATE(p_package != nullptr,
              "[Sam2 Prompt PreProcess] the `package` instance is not a instance of "
              "`Sam2TrackPipelinePackage`!");

  const auto &mode = p_package->current_mode;

  // 0. Get the decoder and encoder buffer
  auto &decoder_blobs_tensor          = p_package->mask_decoder_blobs_buffer;
  auto &memory_attention_blobs_tensor = p_package->memory_attention_blobs_buffer;
  auto &encoder_blobs_tensor          = p_package->image_encoder_blobs_buffer;

  // 1.1
  auto tensor_decoder_high_res_feat_0 =
      decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[1]); // high_res_feat_0 in decoder
  auto tensor_decoder_high_res_feat_1 =
      decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[2]); // high_res_feat_1 in decoder
  auto tensor_decoder_image_embed =
      decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[0]); // image_embed in decoder

  auto tensor_encoder_high_res_feat_0 =
      encoder_blobs_tensor->GetTensor(image_encoder_blob_names_[1]); // hight_res_feat_0 in encoder
  auto tensor_encoder_high_res_feat_1 =
      encoder_blobs_tensor->GetTensor(image_encoder_blob_names_[2]); // hight_res_feat_1 in encoder

  tensor_decoder_high_res_feat_0->ZeroCopy(tensor_encoder_high_res_feat_0);
  tensor_decoder_high_res_feat_1->ZeroCopy(tensor_encoder_high_res_feat_1);

  if (mode == Sam2Mode::Track)
  {
    auto tensor_mem_attn_image_embed = memory_attention_blobs_tensor->GetTensor(
        memory_attention_blob_names_[5]); // image_embed in memory_attention
    tensor_decoder_image_embed->ZeroCopy(tensor_mem_attn_image_embed);
  } else
  {
    auto tensor_encoder_image_embed =
        encoder_blobs_tensor->GetTensor(image_encoder_blob_names_[3]); // image_embed in encoder
    tensor_decoder_image_embed->ZeroCopy(tensor_encoder_image_embed);
  }

  // 2. Set prompt
  const auto &points = p_package->points;
  const auto &labels = p_package->labels;
  const auto &scale  = p_package->transform_scale;
  // 2.1 point coords
  float *points_ptr = decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[3])->Cast<float>();
  // 2.2 point labels
  float *labels_ptr = decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[4])->Cast<float>();
  const uint64_t dynamic_point_number = points.size();
  for (uint64_t i = 0; i < dynamic_point_number; ++i)
  {
    const auto &point     = points[i];
    const auto &lab       = labels[i];
    points_ptr[i * 2 + 0] = static_cast<float>(point.first * scale);
    points_ptr[i * 2 + 1] = static_cast<float>(point.second * scale);
    labels_ptr[i]         = static_cast<float>(lab);
  }

  // 2.3 Set dynamic shape
  std::vector<uint64_t> coords_dynamic_shape{1, dynamic_point_number, 2};
  decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[3])->SetShape(coords_dynamic_shape);
  std::vector<uint64_t> labels_dynamic_shape{1, dynamic_point_number};
  decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[4])->SetShape(labels_dynamic_shape);

  // let unused buffer kept on device side
  decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[6])
      ->SetBufferLocation(DataLocation::DEVICE); // mask for mem
  decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[7])
      ->SetBufferLocation(DataLocation::HOST); // pred_mask

  if (mode == Sam2Mode::Normal)
  {
    decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[5])
        ->SetBufferLocation(DataLocation::DEVICE); // obj_ptr
  } else
  {
    decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[5])
        ->SetBufferLocation(DataLocation::HOST); // obj_ptr
  }
  // 4. Set inference buffer
  p_package->infer_buffer = decoder_blobs_tensor.get();

  return true;
}

bool Sam2::RunMemoryEncoderAndRecord(ParsingType pipeline_unit)
{
  auto p_package = std::dynamic_pointer_cast<Sam2TrackPipelinePackage>(pipeline_unit);
  CHECK_STATE(p_package != nullptr,
              "[Sam2Track] RunMemoryEncoderAndRecord the `package` instance \
                          is not a instance of `SamPipelinePackage`!");

  auto &mask_decoder_blobs_tensor     = p_package->mask_decoder_blobs_buffer;
  auto &memory_attention_blobs_tensor = p_package->memory_attention_blobs_buffer;
  auto &image_encoder_blobs_tensor    = p_package->image_encoder_blobs_buffer;
  auto &memory_encoder_blobs_tensor   = p_package->memory_encoder_blobs_buffer;

  // 1.
  auto tensor_mask_decoder_mask_for_mem = mask_decoder_blobs_tensor->GetTensor(
      image_decoder_blob_names_[6]); // mask_for_mem in mask_decoder
  auto tensor_image_encoder_pixel_feat = image_encoder_blobs_tensor->GetTensor(
      image_encoder_blob_names_[5]); // pixel_feat in image_encoder

  auto tensor_memory_encoder_mask_for_mem = memory_encoder_blobs_tensor->GetTensor(
      memory_encoder_blob_names_[0]); // mask_for_mem in memory_encoder
  auto tensor_memory_encoder_pixel_feat = memory_encoder_blobs_tensor->GetTensor(
      memory_encoder_blob_names_[1]); // pixel_feat in memory_encoder

  tensor_memory_encoder_mask_for_mem->ZeroCopy(tensor_mask_decoder_mask_for_mem);
  tensor_memory_encoder_pixel_feat->ZeroCopy(tensor_image_encoder_pixel_feat);

  MESSURE_DURATION_AND_CHECK_STATE(
      memory_encoder_core_->SyncInfer(memory_encoder_blobs_tensor.get()),
      "[Sam2Track] RunMemoryEncoderAndRecord memory_encoder sync infer failed!!!");

  // 2. record!
  std::vector<float> obj_ptr(OBJ_PTR_DIM);
  std::vector<float> maskmem_feat(MASK_MEM_HEIGHT * MASK_MEM_WIDTH * MASK_MEM_FEAT_DIM);
  std::vector<float> maskmem_pos_enc(MASK_MEM_HEIGHT * MASK_MEM_WIDTH * MASK_MEM_POS_ENC_DIM);
  std::vector<float> time_embeding(MASK_MEM_POOL_SIZE * TIME_EMBEDING_DIM);
  // 3. copy buffer
  memcpy(obj_ptr.data(),
         mask_decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[5])->RawPtr(),
         obj_ptr.size() * sizeof(float));
  memcpy(maskmem_feat.data(),
         memory_encoder_blobs_tensor->GetTensor(memory_encoder_blob_names_[2])->RawPtr(),
         maskmem_feat.size() * sizeof(float));
  memcpy(maskmem_pos_enc.data(),
         memory_encoder_blobs_tensor->GetTensor(memory_encoder_blob_names_[3])->RawPtr(),
         maskmem_pos_enc.size() * sizeof(float));
  memcpy(time_embeding.data(),
         memory_encoder_blobs_tensor->GetTensor(memory_encoder_blob_names_[4])->RawPtr(),
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

bool Sam2::MemoryAttentionPreProcess(ParsingType pipeline_unit)
{
  auto p_package = std::dynamic_pointer_cast<Sam2TrackPipelinePackage>(pipeline_unit);
  CHECK_STATE(p_package != nullptr,
              "[Sam2Track] MemoryAttentionPreProcess the `package` instance \
                          is not a instance of `SamPipelinePackage`!");

  auto &mask_decoder_blobs_tensor     = p_package->mask_decoder_blobs_buffer;
  auto &memory_attention_blobs_tensor = p_package->memory_attention_blobs_buffer;
  auto &image_encoder_blobs_tensor    = p_package->image_encoder_blobs_buffer;
  auto &memory_encoder_blobs_tensor   = p_package->memory_encoder_blobs_buffer;

  // 1. vision_feats & vision_pos_embed
  auto tensor_image_encoder_vision_feat = image_encoder_blobs_tensor->GetTensor(
      image_encoder_blob_names_[3]); // vision_feat in image_encoder
  auto tensor_image_encoder_vision_pos_embed = image_encoder_blobs_tensor->GetTensor(
      image_encoder_blob_names_[4]); // vision_pos_embed in image_encoder

  auto tensor_memory_attention_vision_feat = memory_attention_blobs_tensor->GetTensor(
      memory_attention_blob_names_[0]); // vision_feat in memory_attention
  auto tensor_memory_attention_vision_pos_embed = memory_attention_blobs_tensor->GetTensor(
      memory_attention_blob_names_[1]); // vision_pos_embed in memory_attention

  tensor_memory_attention_vision_feat->ZeroCopy(tensor_image_encoder_vision_feat);
  tensor_image_encoder_vision_pos_embed->ZeroCopy(tensor_memory_attention_vision_pos_embed);

  // Get current obj id
  const auto &memory_dict = memory_bank_[p_package->current_obj_id];

  // 2. memory_0
  float *memory_0_ptr =
      memory_attention_blobs_tensor->GetTensor(memory_attention_blob_names_[2])->Cast<float>();
  size_t memory_0_offset = 0;
  for (const auto &p_id_obj_ptr : memory_dict.obj_ptr_pool_)
  {
    const auto &obj_ptr = p_id_obj_ptr.second;
    memcpy(memory_0_ptr + memory_0_offset, obj_ptr.data(), obj_ptr.size() * sizeof(float));
    memory_0_offset += obj_ptr.size();
  }
  const uint64_t        obj_ptr_num = memory_dict.obj_ptr_pool_.size();
  std::vector<uint64_t> memory_0_shape{obj_ptr_num, 256};
  memory_attention_blobs_tensor->GetTensor(memory_attention_blob_names_[2])
      ->SetShape(memory_0_shape);

  // 3. memory_1
  float *memory_1_ptr =
      memory_attention_blobs_tensor->GetTensor(memory_attention_blob_names_[3])->Cast<float>();
  size_t memory_1_offset = 0;
  for (const auto &p_id_maskmem_feats : memory_dict.maskmem_feats_pool_)
  {
    const auto &maskmem_feats = p_id_maskmem_feats.second;
    memcpy(memory_1_ptr + memory_1_offset, maskmem_feats.data(),
           maskmem_feats.size() * sizeof(float));
    memory_1_offset += maskmem_feats.size();
  }
  const uint64_t        memory_1_num = memory_dict.maskmem_feats_pool_.size();
  std::vector<uint64_t> memory_1_shape{memory_1_num, MASK_MEM_FEAT_DIM, MASK_MEM_HEIGHT,
                                       MASK_MEM_WIDTH};
  memory_attention_blobs_tensor->GetTensor(memory_attention_blob_names_[3])
      ->SetShape(memory_1_shape);

  // 4. memory_pos_embed
  float *memory_pos_embed_ptr =
      memory_attention_blobs_tensor->GetTensor(memory_attention_blob_names_[4])->Cast<float>();
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

  const uint64_t memory_pos_embed_num =
      memory_dict.maskmem_pos_enc_pool_.size() * mask_mem_pixel_count +
      memory_dict.obj_ptr_pool_.size() * 4;
  std::vector<uint64_t> memory_pos_embed_shape{memory_pos_embed_num, 1, MASK_MEM_POS_ENC_DIM};
  memory_attention_blobs_tensor->GetTensor(memory_attention_blob_names_[4])
      ->SetShape(memory_pos_embed_shape);

  // 5. memory attention inference
  memory_attention_blobs_tensor->GetTensor(memory_attention_blob_names_[5])
      ->SetBufferLocation(DataLocation::DEVICE);
  p_package->infer_buffer = memory_attention_blobs_tensor.get();

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
    MESSURE_DURATION_AND_CHECK_STATE(MemoryAttentionPreProcess(current_package_),
                                     "[Sam2-Track] MemoryAttentionPreProcess execute failed!!!");

    MESSURE_DURATION_AND_CHECK_STATE(
        memory_attention_core_->SyncInfer(current_package_->GetInferBuffer()),
        "[Sam2-Track] memory_attention sync infer failed!!!");

    current_package_->points = {{0, 0}};
    current_package_->labels = {-1};
    MESSURE_DURATION_AND_CHECK_STATE(PromptPointPreProcessTrackMode(current_package_),
                                     "[Sam2-Track] Prompt-preprocess execute failed!!!");

    MESSURE_DURATION_AND_CHECK_STATE(
        image_decoder_core_->SyncInfer(current_package_->GetInferBuffer()),
        "[Sam2-Track] image_decoder sync infer failed!!!");

    MESSURE_DURATION_AND_CHECK_STATE(MaskPostProcess(current_package_),
                                     "[Sam2-Track] Mask-postprocess execute failed!!!");

    MESSURE_DURATION_AND_CHECK_STATE(RunMemoryEncoderAndRecord(current_package_),
                                     "[Sam2-Track] Mask-postprocess execute failed!!!");

    results.emplace(obj_id, current_package_->mask);
  }

  return true;
}

} // namespace sam
