#include <cstddef>
#include <iomanip>  // for controlling float print precision
#include <iostream> // for standard I/O
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <print>
#include <sstream> // string to number conversion
#include <string>  // for strings

#include <opencv2/core.hpp>    // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui.hpp> // OpenCV window I/O
#include <opencv2/imgproc.hpp> // Gaussian Blur
#include <opencv2/videoio.hpp>

#include "vp_tree.hpp"

int main(int argc, char *argv[]) {
  VP_tree vp_tree("distances1.csv");

  vp_tree.build();

  // std::println("{}", vp_tree.puntal_search(600));
  vp_tree.print_tree();

  return 0;
}
