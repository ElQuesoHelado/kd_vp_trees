#include <cstddef>
#include <iomanip> // for controlling float print precision
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <print>
#include <sstream> // string to number conversion
#include <string>  // for strings

#include <opencv2/core.hpp>    // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui.hpp> // OpenCV window I/O
#include <opencv2/imgproc.hpp> // Gaussian Blur
#include <opencv2/videoio.hpp>

#include "kd_tree.hpp"
#include "point.hpp"
#include "visual.hpp"
#include "vp_tree.hpp"

int main(int argc, char *argv[]) {
  vector<Point> baseData = readCSV("dataset/images_dataset.csv", 20000, -1);

  VP_tree vp_tree("dataset/distances.csv");
  vp_tree.build();

  KDTree kd_tree;
  kd_tree.build(baseData);

  int id = 1001, n = 10;

  double searchTime;
  auto raw_res_kd = kd_tree.kNearestNeighbors({baseData[id].coords,
                                               baseData[id].id},
                                              n, searchTime);

  auto res_vp = vp_tree.knn(id - 1, n);

  std::vector<int> res_kd;
  for (auto &e : raw_res_kd)
    res_kd.push_back(e.id);

  //   std::print("{} ", e.id);
  // std::println();

  // std::println("{}", res_vp);
  // vp_tree.print_tree();

  // std::println("{}", vp_tree.radial_search(200, 5));
  //
  // auto objs = vp_tree.radial_search(200, 5);

  visual::show_neighbors(id, res_kd);
  // int x;
  // std::cin >> x;
  visual::show_neighbors(id, res_vp);

  return 0;
}
