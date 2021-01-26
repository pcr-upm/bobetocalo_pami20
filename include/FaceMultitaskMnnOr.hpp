/** ****************************************************************************
 *  @file    FaceMultitaskMnnOr.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2020/10
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef FACE_MULTITASK_MNN_OR_HPP
#define FACE_MULTITASK_MNN_OR_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <FaceHeadPose.hpp>
#include <FaceAlignment.hpp>
#include <ShapeCascade.hpp>
#include <opencv2/opencv.hpp>
#include "tensorflow/core/public/session.h"

namespace upm {

/** ****************************************************************************
 * @class FaceMultitaskMnnOr
 * @brief Class used for head pose and facial feature point detection.
 ******************************************************************************/
class FaceMultitaskMnnOr: public FaceHeadPose, public FaceAlignment
{
public:
  FaceMultitaskMnnOr(std::string path) : _path(path) {};

  ~FaceMultitaskMnnOr() {};

  void
  parseOptions
    (
    int argc,
    char **argv
    );

  void
  train
    (
    const std::vector<FaceAnnotation> &anns_train,
    const std::vector<FaceAnnotation> &anns_valid
    );

  void
  load();

  void
  process
    (
    cv::Mat frame,
    std::vector<FaceAnnotation> &faces,
    const FaceAnnotation &ann
    );

private:
  std::string _path;
  std::map< FacePartLabel,std::vector<int> > _cnn_parts;
  std::vector<unsigned int> _cnn_landmarks;
  std::unique_ptr<tensorflow::Session> _session;
  std::vector<ShapeCascade> _sp;
};

} // namespace upm

#endif /* FACE_MULTITASK_MNN_OR_HPP */
