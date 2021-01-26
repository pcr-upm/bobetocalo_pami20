/** ****************************************************************************
 *  @file    face_multitask_bobetocalo_pami20_test.cpp
 *  @brief   Face detection and recognition framework
 *  @author  Roberto Valle Fernandez
 *  @date    2020/10
 *  @copyright All rights reserved.
 *  Software developed by UPM PCR Group: http://www.dia.fi.upm.es/~pcr
 ******************************************************************************/

// ----------------------- INCLUDES --------------------------------------------
#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/shared_ptr.hpp>

#include <trace.hpp>
#include <FaceAnnotation.hpp>
#include <FaceMultitaskMnnOr.hpp>
#include <utils.hpp>

// -----------------------------------------------------------------------------
//
// Purpose and Method:
// Inputs:
// Outputs:
// Dependencies:
// Restrictions and Caveats:
//
// -----------------------------------------------------------------------------
int
main
  (
  int argc,
  char **argv
  )
{
  // Read sample annotations
  upm::FaceAnnotation ann;
  ann.filename = "test/01533.jpg";
  ann.bbox.pos = cv::Rect2f(126.0, 95.0, 119.0, 133.0);
  ann.headpose = cv::Point3f(-10.3706, -1.6728, -8.10081);
  upm::DB_PARTS[upm::FacePartLabel::leyebrow] = {1, 101, 3, 102};
  upm::DB_PARTS[upm::FacePartLabel::reyebrow] = {4, 103, 6, 104};
  upm::DB_PARTS[upm::FacePartLabel::leye] = {7, 9, 8, 10, 105};
  upm::DB_PARTS[upm::FacePartLabel::reye] = {11, 13, 12, 14, 106};
  upm::DB_PARTS[upm::FacePartLabel::nose] = {16, 17, 18, 107};
  upm::DB_PARTS[upm::FacePartLabel::tmouth] = {20, 22, 21, 108};
  upm::DB_PARTS[upm::FacePartLabel::bmouth] = {109, 23};
  upm::DB_PARTS[upm::FacePartLabel::chin] = {24};
  upm::DB_LANDMARKS = {1, 6, 3, 4, 101, 102, 103, 104, 7, 12, 8, 11, 9, 10, 13, 14, 105, 106, 16, 18, 17, 107, 20, 21, 22, 108, 109, 23, 24};
  std::vector<float> coords = {136.364277268,130.66183699,0.0,235.122574115,117.721094644,0.0,176.548687709,122.488736561,0.0,188.12724665,119.083278049,0.0,153.391569828,117.721094644,0.0,155.434844935,125.894195073,0.0,219.45746496,104.780352299,0.0,218.776373258,114.996727835,0.0,146.580652804,140.878212526,0.0,226.949473686,128.618561883,0.0,170.418862388,138.834937418,0.0,199.705805591,134.067295502,0.0,155.434844935,132.705112097,0.0,157.478120042,143.602579335,0.0,216.73309815,125.213103371,0.0,216.73309815,134.748387204,0.0,160.883578554,138.153845716,0.0,215.370914746,130.66183699,0.0,167.013403876,170.846247431,0.0,205.835630912,166.078605514,1.0,180.635237924,164.716422109,0.0,184.040696436,184.468081479,1.0,174.505412602,198.771007229,1.0,218.095281555,194.003365312,1.0,190.170521757,192.641181908,1.0,190.851613459,197.408823824,1.0,191.532705162,199.452098931,1.0,192.894888567,204.900832551,1.0,203.111264103,217.841574896,1.0};
  for (int cont=0; cont < upm::DB_LANDMARKS.size(); cont++)
  {
    unsigned int feature_idx = upm::DB_LANDMARKS[cont];
    float x = coords[(3*cont)+0];
    float y = coords[(3*cont)+1];
    float occluded = coords[(3*cont)+2];
    if (feature_idx < 1)
      continue;
    for (const auto &part : upm::DB_PARTS)
      if (std::find(part.second.begin(),part.second.end(),feature_idx) != part.second.end())
      {
        upm::FaceLandmark landmark;
        landmark.feature_idx = feature_idx;
        landmark.pos = cv::Point2f(x,y);
        landmark.occluded = occluded;
        ann.parts[part.first].landmarks.push_back(landmark);
        break;
      }
  }
  cv::Mat frame = cv::imread(ann.filename, cv::IMREAD_COLOR);
  if (frame.empty())
    return EXIT_FAILURE;

  // Set face detected position
  std::vector<upm::FaceAnnotation> faces(1);
  faces[0].bbox.pos = ann.bbox.pos;

  /// Load face components
  boost::shared_ptr<upm::FaceComposite> composite(new upm::FaceComposite());
  boost::shared_ptr<upm::FaceAlignment> fa(new upm::FaceMultitaskMnnOr("data/"));
  composite->addComponent(fa);

  /// Parse face component options
  composite->parseOptions(argc, argv);
  composite->load();

  // Process frame
  double ticks = processFrame(frame, composite, faces, ann);
  UPM_PRINT("FPS = " << cv::getTickFrequency()/ticks);

  // Evaluate head pose estimation and facial landmark location results
  boost::shared_ptr<upm::FaceHeadPose> fh(new upm::FaceMultitaskMnnOr("data/"));
  composite->addComponent(fh);
  boost::shared_ptr<std::ostream> output(&std::cout, [](std::ostream*){});
  composite->evaluate(output, faces, ann);

  // Draw results
  boost::shared_ptr<upm::Viewer> viewer(new upm::Viewer);
  viewer->init(0, 0, "face_multitask_bobetocalo_pami20_test");
  showResults(viewer, ticks, 0, frame, composite, faces, ann);

  UPM_PRINT("End of face_multitask_bobetocalo_pami20_test");
  return EXIT_SUCCESS;
};
