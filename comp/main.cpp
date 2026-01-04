#include <algorithm>
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

  VP_tree vp_tree(baseData);
  vp_tree.build();

  KDTree kd_tree;
  kd_tree.build(baseData);

  int id = 1100, n = 10;

  auto e = std::find_if(baseData.begin(), baseData.end(),
                        [&](auto &p) { return p.id == id; });

  double searchTime;
  auto raw_res_kd = kd_tree.kNearestNeighbors({e->coords,
                                               e->id},
                                              n, searchTime);

  auto res_vp = vp_tree.knn(id - 1, n);

  std::vector<int> res_kd;
  for (auto &e : raw_res_kd)
    res_kd.push_back(e.id);

  std::println("{}", res_kd);
  std::println("{}", res_vp);

  visual::show_neighbors(id, res_kd);

  visual::show_neighbors(id, res_vp);

  return 0;
}
