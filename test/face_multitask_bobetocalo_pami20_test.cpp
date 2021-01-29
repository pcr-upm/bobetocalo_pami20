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
  ann.filename = "test/01454.jpg";
  ann.bbox.pos = cv::Rect2f(148.0,128.0,114.0,167.0);
  ann.headpose = cv::Point3f(10.4819, 14.7329, -0.278679);
  upm::DB_PARTS[upm::FacePartLabel::leyebrow] = {1, 101, 3, 102};
  upm::DB_PARTS[upm::FacePartLabel::reyebrow] = {4, 103, 6, 104};
  upm::DB_PARTS[upm::FacePartLabel::leye] = {7, 9, 8, 10, 105};
  upm::DB_PARTS[upm::FacePartLabel::reye] = {11, 13, 12, 14, 106};
  upm::DB_PARTS[upm::FacePartLabel::nose] = {16, 17, 18, 107};
  upm::DB_PARTS[upm::FacePartLabel::tmouth] = {20, 22, 21, 108};
  upm::DB_PARTS[upm::FacePartLabel::bmouth] = {109, 23};
  upm::DB_PARTS[upm::FacePartLabel::chin] = {24};
  upm::DB_LANDMARKS = {1, 6, 3, 4, 101, 102, 103, 104, 7, 12, 8, 11, 9, 10, 13, 14, 105, 106, 16, 18, 17, 107, 20, 21, 22, 108, 109, 23, 24};
  std::vector<float> coords = {157.620222307,145.926126478,1.0,248.756324698,150.867240463,1.0,194.953083528,147.024151808,1.0,223.501742108,146.475139143,0.0,176.835665583,138.239949168,1.0,176.286652918,140.435999828,1.0,233.383970078,142.083037823,0.0,234.481995408,144.828101148,0.0,172.992576928,159.102430438,1.0,252.050400689,166.239595083,1.0,196.051108858,164.592557088,1.0,222.952729443,166.239595083,1.0,183.423817563,155.808354448,1.0,182.874804898,166.788607748,1.0,238.325084063,160.200455768,1.0,238.325084063,169.533671073,1.0,182.874804898,161.298481098,1.0,237.776071398,164.592557088,1.0,196.051108858,192.043190338,0.0,228.442856093,195.337266328,0.0,221.854704113,182.160962368,0.0,215.815564798,195.886278993,0.0,192.208020203,221.689874249,0.0,235.580020738,230.474076889,0.0,215.815564798,205.768506964,0.0,215.266552133,212.356658944,0.0,218.560628123,239.258279529,0.0,219.109640788,247.493469504,0.0,218.011615458,284.826330724,0.0};
  std::vector<upm::FacePart> parts = ann.parts;
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
        parts[part.first].landmarks.push_back(landmark);
        break;
      }
  }
  for (const auto &part : upm::DB_PARTS)
    for (int feature_idx : part.second)
      for (const upm::FaceLandmark &landmark : parts[part.first].landmarks)
        if (landmark.feature_idx == feature_idx)
          ann.parts[part.first].landmarks.push_back(landmark);
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
