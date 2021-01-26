/** ****************************************************************************
 *  @file    FaceMultitaskMnnOr.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2020/10
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <FaceMultitaskMnnOr.hpp>
#include <trace.hpp>
#include <utils.hpp>
#include <ModernPosit.h>
#include <transformation.hpp>
#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/vector.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>
#include "tensorflow/cc/ops/standard_ops.h"

namespace upm {

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceMultitaskMnnOr::parseOptions
  (
  int argc,
  char **argv
  )
{
  /// Declare the supported program options
  FaceAlignment::parseOptions(argc, argv);
  namespace po = boost::program_options;
  po::options_description desc("FaceMultitaskMnnOr options");
  UPM_PRINT(desc);
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceMultitaskMnnOr::train
  (
  const std::vector<FaceAnnotation> &anns_train,
  const std::vector<FaceAnnotation> &anns_valid
  )
{
  /// Training CNN model
  UPM_PRINT("Training MNN+OR model");
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceMultitaskMnnOr::load()
{
  UPM_PRINT("Loading MNN model ...");
  std::string trained_model = _path + _database + ".pb";
  tensorflow::GraphDef graph;
  tensorflow::Status load_graph_status = ReadBinaryProto(tensorflow::Env::Default(), trained_model, &graph);
  if (not load_graph_status.ok())
    UPM_ERROR("Failed to load graph: " << trained_model);
  _session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
  tensorflow::Status session_create_status = _session->Create(graph);
  if (not session_create_status.ok())
    UPM_ERROR("Failed to create session");
  if ((_database == "300w_public") or (_database == "300w_private") or (_database == "300wlp") or (_database == "ls3dw") or (_database == "menpo"))
  {
    _cnn_parts[FacePartLabel::leyebrow] = {1, 119, 2, 121, 3};
    _cnn_parts[FacePartLabel::reyebrow] = {4, 124, 5, 126, 6};
    _cnn_parts[FacePartLabel::leye] = {7, 138, 139, 8, 141, 142};
    _cnn_parts[FacePartLabel::reye] = {11, 144, 145, 12, 147, 148};
    _cnn_parts[FacePartLabel::nose] = {128, 129, 130, 17, 16, 133, 134, 135, 18};
    _cnn_parts[FacePartLabel::tmouth] = {20, 150, 151, 22, 153, 154, 21, 165, 164, 163, 162, 161};
    _cnn_parts[FacePartLabel::bmouth] = {156, 157, 23, 159, 160, 168, 167, 166};
    _cnn_parts[FacePartLabel::lear] = {101, 102, 103, 104, 105, 106};
    _cnn_parts[FacePartLabel::rear] = {112, 113, 114, 115, 116, 117};
    _cnn_parts[FacePartLabel::chin] = {107, 108, 24, 110, 111};
    _cnn_landmarks = {101, 102, 103, 104, 105, 106, 107, 108, 24, 110, 111, 112, 113, 114, 115, 116, 117, 1, 119, 2, 121, 3, 4, 124, 5, 126, 6, 128, 129, 130, 17, 16, 133, 134, 135, 18, 7, 138, 139, 8, 141, 142, 11, 144, 145, 12, 147, 148, 20, 150, 151, 22, 153, 154, 21, 156, 157, 23, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168};
  }
  else if (_database == "cofw")
  {
    _cnn_parts[FacePartLabel::leyebrow] = {1, 101, 3, 102};
    _cnn_parts[FacePartLabel::reyebrow] = {4, 103, 6, 104};
    _cnn_parts[FacePartLabel::leye] = {7, 9, 8, 10, 105};
    _cnn_parts[FacePartLabel::reye] = {11, 13, 12, 14, 106};
    _cnn_parts[FacePartLabel::nose] = {16, 17, 18, 107};
    _cnn_parts[FacePartLabel::tmouth] = {20, 22, 21, 108};
    _cnn_parts[FacePartLabel::bmouth] = {109, 23};
    _cnn_parts[FacePartLabel::chin] = {24};
    _cnn_landmarks = {1, 101, 3, 102, 4, 103, 6, 104, 7, 9, 8, 10, 105, 11, 13, 12, 14, 106, 16, 17, 18, 107, 20, 22, 21, 108, 109, 23, 24};
  }
  else if (_database == "aflw")
  {
    _cnn_parts[FacePartLabel::leyebrow] = {1, 2, 3};
    _cnn_parts[FacePartLabel::reyebrow] = {4, 5, 6};
    _cnn_parts[FacePartLabel::leye] = {7, 101, 8};
    _cnn_parts[FacePartLabel::reye] = {11, 102, 12};
    _cnn_parts[FacePartLabel::nose] = {16, 17, 18};
    _cnn_parts[FacePartLabel::tmouth] = {20, 103, 21};
    _cnn_parts[FacePartLabel::lear] = {15};
    _cnn_parts[FacePartLabel::rear] = {19};
    _cnn_parts[FacePartLabel::chin] = {24};
    _cnn_landmarks = {1, 2, 3, 4, 5, 6, 7, 101, 8, 11, 102, 12, 15, 16, 17, 18, 19, 20, 103, 21, 24};
  }
  else if (_database == "wflw")
  {
    _cnn_parts[FacePartLabel::leyebrow] = {1, 134, 2, 136, 3, 138, 139, 140, 141};
    _cnn_parts[FacePartLabel::reyebrow] = {6, 147, 148, 149, 150, 4, 143, 5, 145};
    _cnn_parts[FacePartLabel::leye] = {7, 161, 9, 163, 8, 165, 10, 167, 196};
    _cnn_parts[FacePartLabel::reye] = {11, 169, 13, 171, 12, 173, 14, 175, 197};
    _cnn_parts[FacePartLabel::nose] = {151, 152, 153, 17, 16, 156, 157, 158, 18};
    _cnn_parts[FacePartLabel::tmouth] = {20, 177, 178, 22, 180, 181, 21, 192, 191, 190, 189, 188};
    _cnn_parts[FacePartLabel::bmouth] = {187, 186, 23, 184, 183, 193, 194, 195};
    _cnn_parts[FacePartLabel::lear] = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110};
    _cnn_parts[FacePartLabel::rear] = {122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132};
    _cnn_parts[FacePartLabel::chin] = {111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121};
    _cnn_landmarks = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 1, 134, 2, 136, 3, 138, 139, 140, 141, 4, 143, 5, 145, 6, 147, 148, 149, 150, 151, 152, 153, 17, 16, 156, 157, 158, 18, 7, 161, 9, 163, 8, 165, 10, 167, 11, 169, 13, 171, 12, 173, 14, 175, 20, 177, 178, 22, 180, 181, 21, 183, 184, 23, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197};
  }
  else if (_database == "3dmenpo")
  {
    _cnn_parts[upm::FacePartLabel::leyebrow] = {1, 134, 2, 136, 3};
    _cnn_parts[upm::FacePartLabel::reyebrow] = {4, 139, 5, 141, 6};
    _cnn_parts[upm::FacePartLabel::leye] = {7, 153, 154, 8, 156, 157};
    _cnn_parts[upm::FacePartLabel::reye] = {11, 159, 160, 12, 162, 163};
    _cnn_parts[upm::FacePartLabel::nose] = {143, 144, 145, 17, 16, 148, 149, 150, 18};
    _cnn_parts[upm::FacePartLabel::tmouth] = {20, 165, 166, 22, 168, 169, 21, 180, 179, 178, 177, 176};
    _cnn_parts[upm::FacePartLabel::bmouth] = {171, 172, 23, 174, 175, 183, 182, 181};
    _cnn_parts[upm::FacePartLabel::lear] = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110};
    _cnn_parts[upm::FacePartLabel::rear] = {122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132};
    _cnn_parts[upm::FacePartLabel::chin] = {111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121};
    _cnn_landmarks = {100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 24, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 1, 134, 2, 136, 3, 4, 139, 5, 141, 6, 143, 144, 145, 17, 16, 148, 149, 150, 18, 7, 153, 154, 8, 156, 157, 11, 159, 160, 12, 162, 163, 20, 165, 166, 22, 168, 169, 21, 171, 172, 23, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183};
  }
  else
  {
    _cnn_parts[FacePartLabel::leyebrow] = {1, 2, 3};
    _cnn_parts[FacePartLabel::reyebrow] = {4, 5, 6};
    _cnn_parts[FacePartLabel::leye] = {7, 9, 8, 10};
    _cnn_parts[FacePartLabel::reye] = {11, 13, 12, 14};
    _cnn_parts[FacePartLabel::nose] = {16, 17, 18};
    _cnn_parts[FacePartLabel::tmouth] = {20, 22, 21};
    _cnn_parts[FacePartLabel::bmouth] = {23};
    _cnn_parts[FacePartLabel::lear] = {15};
    _cnn_parts[FacePartLabel::rear] = {19};
    _cnn_parts[FacePartLabel::chin] = {24};
    _cnn_landmarks = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
  }
  UPM_PRINT("Loading OR model ...");
  std::vector<std::string> paths;
  boost::filesystem::path dir_path(_path + _database + "/");
  boost::filesystem::directory_iterator end_it;
  for (boost::filesystem::directory_iterator it(dir_path); it != end_it; ++it)
    paths.push_back(it->path().string());
  sort(paths.begin(), paths.end());
  _sp.resize(HP_LABELS.size());
  for (const std::string &path : paths)
  {
    try
    {
      std::ifstream ifs(path);
      cereal::BinaryInputArchive ia(ifs);
      ia >> _sp[boost::lexical_cast<unsigned int>(path.substr(0,path.size()-4).substr(path.find_last_of('_')+1))] >> DB_PARTS >> DB_LANDMARKS;
      ifs.close();
    }
    catch (cereal::Exception &ex)
    {
      UPM_ERROR("Exception during predictor deserialization: " << ex.what());
    }
  }
};

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
void
FaceMultitaskMnnOr::process
  (
  cv::Mat frame,
  std::vector<FaceAnnotation> &faces,
  const FaceAnnotation &ann
  )
{
  /// Analyze each detected face
  for (FaceAnnotation &face : faces)
  {
    int yaw_idx = getHeadposeIdx(FaceAnnotation().headpose.x);
    _sp[yaw_idx].process(frame, face, ann, _measure, _path, _cnn_parts, _cnn_landmarks, _session);
  }
};

} // namespace upm
