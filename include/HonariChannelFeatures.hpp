/** ****************************************************************************
 *  @file    HonariChannelFeatures.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2020/10
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef HONARI_CHANNEL_FEATURES_HPP
#define HONARI_CHANNEL_FEATURES_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <ChannelFeatures.hpp>
#include <FaceAnnotation.hpp>
#include <FeaturesRelativeEncoding.hpp>
#include <cereal/access.hpp>
#include <cereal/types/polymorphic.hpp>
#include <opencv2/opencv.hpp>
#include "tensorflow/core/public/session.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace upm {

/** ****************************************************************************
 * @class HonariChannelFeatures
 * @brief Computes maximum probability from the heat map channels.
 ******************************************************************************/
class HonariChannelFeatures : public ChannelFeatures
{
public:
  HonariChannelFeatures() {};

  HonariChannelFeatures
    (
    const cv::Mat &shape,
    const cv::Mat &label,
    const std::string &path,
    const std::string &database
    );

  virtual
  ~HonariChannelFeatures() {};

  cv::Rect_<float>
  enlargeBbox
    (
    const cv::Rect_<float> &bbox
    ) override;

  tensorflow::Status
  imageToTensor
    (
    const cv::Mat &img,
    std::vector<tensorflow::Tensor>* output_tensors
    );

  std::vector<cv::Mat>
  tensorToMaps
    (
    const tensorflow::Tensor &img_tensor,
    const cv::Size &face_size
    );

  std::vector<float>
  tensorToVector
    (
    const tensorflow::Tensor &img_tensor
    );

  void
  loadChannelsGenerator() {};

  std::vector<cv::Mat>
  generateChannels
    (
    const cv::Mat &img,
    const cv::Rect_<float> &bbox
    ) {return {};};

  std::vector<cv::Mat>
  generateChannels
    (
    const cv::Mat &img,
    const cv::Rect_<float> &bbox,
    const std::map< FacePartLabel,std::vector<int> > &cnn_parts,
    const std::vector<unsigned int> &cnn_landmarks,
    std::unique_ptr<tensorflow::Session> &session,
    std::vector<float> &pose_euler,
    std::vector<float> &pose_proj,
    std::vector<float> &visibilities
    );

  void
  loadFeaturesDescriptor() {};

  cv::Mat
  extractFeatures
    (
    const std::vector<cv::Mat> &img_channels,
    const float face_height,
    const cv::Mat &rigid,
    const cv::Mat &tform,
    const cv::Mat &shape,
    float level
    ) override;

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned version)
  {
    ar & map_scale & _max_diameter & _min_diameter  & _robust_shape & _robust_label & _sampling_pattern & _encoder & _path & _database;
  };

  float map_scale;

private:
  float _max_diameter, _min_diameter;
  cv::Mat _robust_shape, _robust_label;
  std::vector<cv::Point2f> _sampling_pattern;
  std::shared_ptr<FeaturesRelativeEncoding> _encoder;
  std::string _path;
  std::string _database;
};

} // namespace upm

CEREAL_REGISTER_TYPE(upm::HonariChannelFeatures);
CEREAL_REGISTER_POLYMORPHIC_RELATION(upm::ChannelFeatures, upm::HonariChannelFeatures);

#endif /* HONARI_CHANNEL_FEATURES_HPP */
