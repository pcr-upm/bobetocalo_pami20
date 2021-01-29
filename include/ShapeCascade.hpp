/** ****************************************************************************
 *  @file    ShapeCascade.hpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2020/10
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ------------------ RECURSION PROTECTION -------------------------------------
#ifndef SHAPE_CASCADE_HPP
#define SHAPE_CASCADE_HPP

// ----------------------- INCLUDES --------------------------------------------
#include <utils.hpp>
#include <FaceAnnotation.hpp>
#include <ModernPosit.h>
#include <ChannelFeatures.hpp>
#include <HonariChannelFeatures.hpp>
#include <LearningAlgorithm.hpp>
#include <transformation.hpp>
#include <cereal/access.hpp>
#include <cereal/types/vector.hpp>
#include <serialization.hpp>
#include <opencv2/opencv.hpp>
#include <numeric>
#include <fstream>

namespace upm {

enum class ChannelFeature { honari };
enum class InitialShape { honari };
enum class RuntimeMode { train, test };
const unsigned int MAX_NUM_LEVELS = 50;

/** ****************************************************************************
 * @class ShapeCascade
 * @brief Use cascade of regression trees that can localize facial landmarks.
 ******************************************************************************/
class ShapeCascade
{
public:
  /// Train predictor
  ShapeCascade
    (
    const cv::Size2f &shape_size,
    const cv::Mat &robust_shape,
    const cv::Mat &robust_label,
    const std::vector<cv::Mat> &initial_shapes,
    const std::vector<cv::Mat> &initial_labels,
    const std::vector<LearningAlgorithm::EnsembleTrees> &forests,
    const std::vector<std::vector<std::vector<int>>> &landmarks,
    const std::string &path,
    const std::string &database,
    const ChannelFeature &feature_mode,
    const InitialShape &initial_mode,
    const int &feats_convergence_iter
    )
  {
    _shape_size = shape_size;
    _robust_shape = robust_shape.clone();
    _robust_label = robust_label.clone();
    _initial_shapes = initial_shapes;
    _initial_labels = initial_labels;
    _forests = forests;
    _landmarks = landmarks;
    _feature_mode = feature_mode;
    _initial_mode = initial_mode;
    _feats_convergence_iter = feats_convergence_iter;
    _cf.reset(new HonariChannelFeatures(robust_shape, robust_label, path, database));
    _hcf.reset(new HonariChannelFeatures(robust_shape, robust_label, path, database));
  };

  /// Test predictor
  ShapeCascade() {};

  ~ShapeCascade() {};

  void
  process
    (
    const cv::Mat &img,
    FaceAnnotation &face,
    const FaceAnnotation &ann,
    const ErrorMeasure &measure,
    const std::string &path,
    const std::map< FacePartLabel,std::vector<int> > &cnn_parts,
    const std::vector<unsigned int> &cnn_landmarks,
    std::unique_ptr<tensorflow::Session> &session
    )
  {
    /// Map shape from normalized space into the image dimension
    const auto num_landmarks = static_cast<unsigned int>(_robust_shape.rows);
    cv::Mat utform = unnormalizingTransform(face.bbox.pos, _shape_size);

    /// Load MNN channels into memory only once
    std::shared_ptr<HonariChannelFeatures> hcf(_hcf);
    cv::Rect_<float> bbox = hcf->enlargeBbox(face.bbox.pos);
    std::vector<float> poses_ed, poses_ae, visibilities;
    std::vector<cv::Mat> channels = hcf->generateChannels(img, bbox, cnn_parts, cnn_landmarks, session, poses_ed, poses_ae, visibilities);
    face.headpose = cv::Point3f(poses_ed[0], poses_ed[1], poses_ed[2]);

    /// Set bbox according to cropped channel features
    cv::Rect_<float> feat_bbox = bbox;
    cv::Point2f feat_scale = cv::Point2f(channels[0].cols/feat_bbox.width, channels[0].rows/feat_bbox.height);
    feat_scale *= hcf->map_scale;
    feat_bbox = cv::Rect_<float>(face.bbox.pos.x-feat_bbox.x, face.bbox.pos.y-feat_bbox.y, face.bbox.pos.width, face.bbox.pos.height);
    feat_bbox.x *= feat_scale.x;
    feat_bbox.y *= feat_scale.y;
    feat_bbox.width *= feat_scale.x;
    feat_bbox.height *= feat_scale.y;
    cv::Mat feat_utform = unnormalizingTransform(feat_bbox, _shape_size);

    /// Run algorithm several times using different initializations
    const unsigned int num_initial_shapes = static_cast<int>(_initial_shapes.size());
    std::vector<cv::Mat> current_shapes(num_initial_shapes), current_labels(num_initial_shapes), current_rigids(num_initial_shapes);
    for (unsigned int i=0; i < num_initial_shapes; i++)
    {
      cv::RNG rnd = cv::RNG();
      FaceAnnotation initial_face = generateInitialShape(path, RuntimeMode::test, cnn_landmarks, bbox, poses_ae, visibilities, hcf->map_scale, rnd);
      const cv::Mat ntform = normalizingTransform(face.bbox.pos, _shape_size);
      current_shapes[i] = cv::Mat::zeros(num_landmarks,3,CV_32FC1);
      current_labels[i] = cv::Mat::zeros(num_landmarks,1,CV_32FC1);
      facePartsToShape(initial_face.parts, ntform, 1.0f, current_shapes[i], current_labels[i]);
    }

    for (unsigned int i=0; i < _forests.size(); i++)
      for (unsigned int j=0; j < num_initial_shapes; j++)
      {
        /// Global similarity transform that maps 'robust_shape' to 'current_shape'
        float level = static_cast<float>(i) / static_cast<float>(MAX_NUM_LEVELS);
        current_rigids[j] = findSimilarityTransform(_robust_shape, current_shapes[j], current_labels[j]);
        cv::Mat features = hcf->extractFeatures(channels, face.bbox.pos.height, current_rigids[j], feat_utform, current_shapes[j], level);
        /// Update sample using a mean residual
        cv::Mat mean_residual = cv::Mat::zeros(num_landmarks,3,CV_32FC1);
        for (const auto &trees : _forests[i])
          for (const auto &tree : trees)
            mean_residual += tree.leafs[tree.predict(features)].residual;
        addResidualToShape(mean_residual, current_shapes[j]);
      }
    /// Facial feature location obtained
    bestEstimation(current_shapes, current_labels, utform, ann, measure, face);
  };

static FaceAnnotation
  generateInitialShape
    (
    const std::string &path,
    const RuntimeMode &runtime_mode,
    const std::vector<unsigned int> &cnn_landmarks,
    const cv::Rect_<float> &bbox,
    const std::vector<float> &poses,
    const std::vector<float> &visibilities,
    const float &map_scale,
    cv::RNG &rnd
    )
  {
    /// Load 3D face shape
    std::vector<cv::Point3f> world_all;
    std::vector<unsigned int> index_all;
    ModernPosit::loadWorldShape(path + "../../../headpose/posit/data/", DB_LANDMARKS, world_all, index_all);
    /// Intrinsic parameters
    cv::Rect_<float> bbox_cnn = cv::Rect_<float>(0, 0, bbox.width, bbox.height);
    double focal_length = static_cast<double>(bbox_cnn.width) * 1.5;
    cv::Point2f face_center = (bbox_cnn.tl() + bbox_cnn.br()) * 0.5f;
    cv::Mat cam_matrix = (cv::Mat_<float>(3,3) << focal_length,0,face_center.x, 0,focal_length,face_center.y, 0,0,1);
    /// Extrinsic parameters
    cv::Mat rot_matrix, trl_matrix;
    cv::Point3f headpose = cv::Point3f(poses[0], poses[1], poses[2]);
    rot_matrix = ModernPosit::eulerToRotationMatrix(headpose);
    trl_matrix = (cv::Mat_<float>(3,1) << poses[3],poses[4],poses[5]);
    if (runtime_mode == RuntimeMode::train)
    {
      cv::Point3f headpose_train = ModernPosit::rotationMatrixToEuler(rot_matrix);
      headpose_train += cv::Point3f(rnd.uniform(-20.0f,20.0f),rnd.uniform(-10.0f,10.0f),rnd.uniform(-10.0f,10.0f));
      rot_matrix = ModernPosit::eulerToRotationMatrix(headpose_train);
    }
    /// Project 3D shape into 2D landmarks
    cv::Mat rot_vector;
    cv::Rodrigues(rot_matrix, rot_vector);
    std::vector<cv::Point2f> image_all_proj;
    cv::projectPoints(world_all, rot_vector, trl_matrix, cam_matrix, cv::Mat(), image_all_proj);
    headpose = ModernPosit::rotationMatrixToEuler(rot_matrix);
    FaceAnnotation initial_face;
    for (const auto &db_part : DB_PARTS)
      for (int feature_idx : db_part.second)
      {
        FaceLandmark landmark;
        landmark.feature_idx = static_cast<unsigned int>(feature_idx);
        auto shape_idx = static_cast<unsigned int>(std::distance(index_all.begin(),std::find(index_all.begin(),index_all.end(),feature_idx)));
        landmark.pos.x = (bbox.x + image_all_proj[shape_idx].x);
        landmark.pos.y = (bbox.y + image_all_proj[shape_idx].y);
        shape_idx = static_cast<unsigned int>(std::distance(cnn_landmarks.begin(), std::find(cnn_landmarks.begin(),cnn_landmarks.end(),feature_idx)));
        landmark.occluded = setSelfOcclusion(headpose, landmark.feature_idx) ? 1.0f : 1.0f-visibilities[shape_idx];
        initial_face.parts[db_part.first].landmarks.push_back(landmark);
      }
    return initial_face;
  };

  static bool
  setSelfOcclusion
    (
    const cv::Point3f &headpose,
    const int &feature_idx
    )
  {
    bool occluded = false;
    if ((std::find(DB_PARTS[FacePartLabel::lear].begin(),DB_PARTS[FacePartLabel::lear].end(),feature_idx) != DB_PARTS[FacePartLabel::lear].end()) and (headpose.x < -15.0f))
      occluded = true;
    else if ((std::find(DB_PARTS[FacePartLabel::rear].begin(),DB_PARTS[FacePartLabel::rear].end(),feature_idx) != DB_PARTS[FacePartLabel::rear].end()) and (headpose.x > 15.0f))
      occluded = true;
    else if ((std::find(DB_PARTS[FacePartLabel::leye].begin(),DB_PARTS[FacePartLabel::leye].end(),feature_idx) != DB_PARTS[FacePartLabel::leye].end() or (std::find(DB_PARTS[FacePartLabel::leyebrow].begin(),DB_PARTS[FacePartLabel::leyebrow].end(),feature_idx) != DB_PARTS[FacePartLabel::leyebrow].end())) and (headpose.x < -40.0f))
      occluded = true;
    else if ((std::find(DB_PARTS[FacePartLabel::reye].begin(),DB_PARTS[FacePartLabel::reye].end(),feature_idx) != DB_PARTS[FacePartLabel::reye].end() or (std::find(DB_PARTS[FacePartLabel::reyebrow].begin(),DB_PARTS[FacePartLabel::reyebrow].end(),feature_idx) != DB_PARTS[FacePartLabel::reyebrow].end())) and (headpose.x > 40.0f))
      occluded = true;
    return occluded;
  }

  static void
  bestEstimation
    (
    const std::vector<cv::Mat> &shapes,
    const std::vector<cv::Mat> &labels,
    const cv::Mat &tform,
    const FaceAnnotation &ann,
    const ErrorMeasure &measure,
    FaceAnnotation &face
    )
  {
    unsigned int best_idx = 0;
    float err, best_err = std::numeric_limits<float>::max();
    const auto num_initials = static_cast<unsigned int>(shapes.size());
    for (unsigned int i=0; i < num_initials; i++)
    {
      FaceAnnotation current;
      shapeToFaceParts(shapes[i], labels[i], tform, 1.0f, current.parts);
      std::vector<unsigned int> indices;
      std::vector<float> errors;
      getNormalizedErrors(current, ann, measure, indices, errors);
      err = static_cast<float>(std::accumulate(errors.begin(),errors.end(),0.0)) / static_cast<float>(errors.size());
      if (err < best_err)
      {
        best_err = err;
        best_idx = i;
      }
    }
    shapeToFaceParts(shapes[best_idx], labels[best_idx], tform, 1.0f, face.parts);
  };

  friend class cereal::access;
  template<class Archive>
  void serialize(Archive &ar, const unsigned version)
  {
    ar & _shape_size & _robust_shape & _robust_label & _initial_shapes & _initial_labels & _forests & _landmarks & _initial_mode & _feature_mode & _feats_convergence_iter & _cf & _hcf;
  };

private:
  cv::Size2f _shape_size;
  cv::Mat _robust_shape;
  cv::Mat _robust_label;
  std::vector<cv::Mat> _initial_shapes;
  std::vector<cv::Mat> _initial_labels;
  std::vector<LearningAlgorithm::EnsembleTrees> _forests;
  std::vector<std::vector<std::vector<int>>> _landmarks;
  InitialShape _initial_mode;
  ChannelFeature _feature_mode;
  int _feats_convergence_iter;
  std::shared_ptr<ChannelFeatures> _cf;
  std::shared_ptr<HonariChannelFeatures> _hcf;
};

} // namespace upm

#endif /* SHAPE_CASCADE_HPP */
