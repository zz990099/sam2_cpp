#include "sam2_def.hpp"

#include "deploy_core/wrapper.h"

namespace sam {

static void ThrowRuntimeError(const std::string &hint, uint64_t line_num)
{
  std::string exception_message = "[Sam2:" + std::to_string(line_num) + "] " + hint;
  throw std::runtime_error(exception_message);
}

static void CheckBlobNameMatched(const std::string &infer_core_name,
                                 const std::shared_ptr<inference_core::BaseInferCore> &infer_core,
                                 const std::vector<std::string>                       &blob_names)
{
  auto blobs_tensor = infer_core->AllocBlobsBuffer();
  if (blob_names.size() != blobs_tensor->Size())
  {
    ThrowRuntimeError(infer_core_name + " core got different blob size with blob_names input! " +
                          std::to_string(blobs_tensor->Size()) + " vs " +
                          std::to_string(blob_names.size()),
                      __LINE__);
  }
  for (const auto &blob_name : blob_names)
  {
    try
    {
      auto buffer_ptr = blobs_tensor->GetTensor(blob_name);
    } catch (std::exception e)
    {
      ThrowRuntimeError(infer_core_name + " met invalid blob_name in blob_names : " + blob_name,
                        __LINE__);
    }
  }
}

const std::string Sam2::model_name_ = "sam2";

Sam2::Sam2(std::shared_ptr<inference_core::BaseInferCore>      image_encoder_core,
           std::shared_ptr<inference_core::BaseInferCore>      image_decoder_core,
           std::shared_ptr<inference_core::BaseInferCore>      memory_attention_core,
           std::shared_ptr<inference_core::BaseInferCore>      memory_encoder_core,
           std::shared_ptr<detection_2d::IDetectionPreProcess> image_preprocess_block,
           const std::vector<std::string>                     &image_encoder_blob_names,
           const std::vector<std::string>                     &image_dec_blob_names,
           const std::vector<std::string>                     &memory_attention_blob_names,
           const std::vector<std::string>                     &memory_encoder_blob_names)
    : BaseSam2TrackModel(model_name_, image_encoder_core, image_decoder_core, image_decoder_core),
      image_encoder_core_(image_encoder_core),
      image_decoder_core_(image_decoder_core),
      memory_attention_core_(memory_attention_core),
      memory_encoder_core_(memory_encoder_core),
      image_preprocess_block_(image_preprocess_block),
      image_encoder_blob_names_(image_encoder_blob_names),
      image_decoder_blob_names_(image_dec_blob_names),
      memory_attention_blob_names_(memory_attention_blob_names),
      memory_encoder_blob_names_(memory_encoder_blob_names)
{
  // Check
  CheckBlobNameMatched("image_encoder", image_encoder_core_, image_encoder_blob_names_);
  CheckBlobNameMatched("image_decoder", image_decoder_core_, image_decoder_blob_names_);
  CheckBlobNameMatched("memory_attention", memory_attention_core_, memory_attention_blob_names_);
  CheckBlobNameMatched("memory_encoder", memory_encoder_core_, memory_encoder_blob_names_);

  if (image_preprocess_block_ == nullptr)
  {
    throw std::invalid_argument("[Sam2] Got INVALID preprocess_block ptr!!!");
  }
}

bool Sam2::ImagePreProcess(ParsingType package)
{
  auto p_package = std::dynamic_pointer_cast<SamPipelinePackage>(package);
  CHECK_STATE(p_package != nullptr,
              "[Sam2 Image PreProcess] the `package` instance \
                                    is not a instance of `SamPipelinePackage`!");

  auto encoder_blobs_buffer = p_package->image_encoder_blobs_buffer;
  // make the output buffer at device side
  // (some inference framework will still output buffer to host side)
  encoder_blobs_buffer->GetTensor(image_encoder_blob_names_[1])
      ->SetBufferLocation(DataLocation::DEVICE);
  encoder_blobs_buffer->GetTensor(image_encoder_blob_names_[2])
      ->SetBufferLocation(DataLocation::DEVICE);
  encoder_blobs_buffer->GetTensor(image_encoder_blob_names_[3])
      ->SetBufferLocation(DataLocation::DEVICE);
  encoder_blobs_buffer->GetTensor(image_encoder_blob_names_[4])
      ->SetBufferLocation(DataLocation::DEVICE);
  encoder_blobs_buffer->GetTensor(image_encoder_blob_names_[5])
      ->SetBufferLocation(DataLocation::DEVICE);

  // preprocess image and write into buffer
  const auto scale = image_preprocess_block_->Preprocess(
      p_package->input_image_data, encoder_blobs_buffer->GetTensor(image_encoder_blob_names_[0]),
      IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH);
  // set the inference buffer
  p_package->infer_buffer = encoder_blobs_buffer.get();
  // record transform factor
  p_package->transform_scale = scale;

  return true;
}

bool Sam2::PromptBoxPreProcess(ParsingType package)
{
  auto p_package = std::dynamic_pointer_cast<SamPipelinePackage>(package);
  CHECK_STATE(p_package != nullptr,
              "[Sam2 Prompt PreProcess] the `package` instance \
                          is not a instance of `SamPipelinePackage`!");

  const auto &boxes  = p_package->boxes;
  auto       &points = p_package->points;
  auto       &labels = p_package->labels;
  const auto &scale  = p_package->transform_scale;
  for (const auto &box : boxes)
  {
    int x0 = box.x - box.w / 2;
    int y0 = box.y - box.h / 2;
    int x1 = box.x + box.w / 2;
    int y1 = box.y + box.h / 2;
    points.push_back({x0, y0});
    points.push_back({x1, y1});
    labels.push_back(2);
    labels.push_back(3);
  }

  return PromptPointPreProcess(package);
}


bool Sam2::PromptPointPreProcess(ParsingType package)
{
  auto p_package = std::dynamic_pointer_cast<SamPipelinePackage>(package);
  CHECK_STATE(p_package != nullptr,
              "[Sam2 Prompt PreProcess] the `package` instance is not a instance of "
              "`SamPipelinePackage`!");

  // 0. Get the decoder and encoder buffer
  auto &decoder_blobs_tensor          = p_package->mask_decoder_blobs_buffer;
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

  auto tensor_encoder_image_embed =
      encoder_blobs_tensor->GetTensor(image_encoder_blob_names_[3]); // image_embed in encoder
  tensor_decoder_image_embed->ZeroCopy(tensor_encoder_image_embed);

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
  decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[5])
      ->SetBufferLocation(DataLocation::DEVICE); // obj_ptr

  // 4. Set inference buffer
  p_package->infer_buffer = decoder_blobs_tensor.get();

  return true;
}

bool Sam2::MaskPostProcess(ParsingType package)
{
  auto p_package = std::dynamic_pointer_cast<SamPipelinePackage>(package);
  CHECK_STATE(p_package != nullptr,
              "[Sam2 Mask PostProcess] the `package` instance \
                          is not a instance of `SamPipelinePackage`!");

  auto decoder_blobs_tensor = p_package->mask_decoder_blobs_buffer;

  // 1. Get the output masks buffer
  void *decoder_output_masks_ptr =
      decoder_blobs_tensor->GetTensor(image_decoder_blob_names_[7])->RawPtr(); // pred_mask
  cv::Mat masks_output(IMAGE_INPUT_HEIGHT, IMAGE_INPUT_WIDTH, CV_32FC1, decoder_output_masks_ptr);

  // 2. crop valid block
  const auto &input_image_info = p_package->input_image_data->GetImageDataInfo();
  const auto &scale            = p_package->transform_scale;
  masks_output                 = masks_output(cv::Range(0, input_image_info.image_height * scale),
                                              cv::Range(0, input_image_info.image_width * scale));

  // 3. resize to original size
  cv::resize(masks_output, masks_output,
             {input_image_info.image_width, input_image_info.image_height}, 0, 0, cv::INTER_LINEAR);

  // 4. convert to binary mask
  cv::threshold(masks_output, masks_output, 0, 255, cv::THRESH_BINARY);

  // 5. convert to CV_8U
  masks_output.convertTo(masks_output, CV_8U);

  p_package->mask = masks_output;

  return true;
}

std::shared_ptr<BaseSam2TrackModel> CreateSam2Model(
    std::shared_ptr<inference_core::BaseInferCore>      image_encoder_core,
    std::shared_ptr<inference_core::BaseInferCore>      image_decoder_core,
    std::shared_ptr<inference_core::BaseInferCore>      memory_attention_core,
    std::shared_ptr<inference_core::BaseInferCore>      memory_encoder_core,
    std::shared_ptr<detection_2d::IDetectionPreProcess> image_preprocess_block,
    const std::vector<std::string>                     &image_encoder_blob_names,
    const std::vector<std::string>                     &image_dec_blob_names,
    const std::vector<std::string>                     &memory_attention_blob_names,
    const std::vector<std::string>                     &memory_encoder_blob_names)
{
  return std::make_shared<Sam2>(image_encoder_core, image_decoder_core, memory_attention_core,
                                memory_encoder_core, image_preprocess_block,
                                image_encoder_blob_names, image_dec_blob_names,
                                memory_attention_blob_names, memory_encoder_blob_names);
}

} // namespace sam