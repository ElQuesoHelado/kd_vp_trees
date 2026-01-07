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

#include "funcs.hpp"

void wait_until_close(const std::vector<std::string> &windows) {
  bool initialized = false;

  while (true) {
    int key = cv::waitKey(30);
    if (key == 27)
      break; // ESC

    bool any_alive = false;

    for (const auto &w : windows) {
      double v = cv::getWindowProperty(w, cv::WND_PROP_VISIBLE);

      if (v >= 1) {
        any_alive = true;
        initialized = true;
        break;
      }
    }

    if (initialized && !any_alive)
      break;
  }
}

int main() {
  auto baseData = readCSV("dataset/images_dataset.csv", 20000, -1);

  VP_tree vp_tree(baseData);
  vp_tree.build();

  KDTree kd_tree;
  kd_tree.build(baseData);

  while (true) {
    int id, k;

    std::cout << "\nID de imagen (-1 para salir): ";
    std::cin >> id;
    if (id < 0)
      break;

    std::cout << "k vecinos: ";
    std::cin >> k;

    auto it = std::find_if(baseData.begin(), baseData.end(),
                           [&](auto &p) { return p.id == id; });

    if (it == baseData.end()) {
      std::cout << "ID no encontrado\n";
      continue;
    }

    double searchTime;
    auto raw_kd = kd_tree.kNearestNeighbors({it->coords, it->id},
                                            k, searchTime);

    auto res_vp = vp_tree.knn(id, k);

    std::vector<int> res_kd;
    for (auto &e : raw_kd)
      res_kd.push_back(e.id);

    std::vector<std::string> windows;

    windows.push_back("KD-tree");
    visual::show_neighbors(id, res_kd, 0, "KD-tree");

    windows.push_back("VP-tree");
    visual::show_neighbors(id, res_vp, 1, "VP-tree");

    std::cout << "ESC o cerrar ventanas para continuar...\n";
    wait_until_close(windows);

    cv::destroyAllWindows();
  }

  return 0;
}
