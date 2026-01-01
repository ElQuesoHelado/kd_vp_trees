#pragma once

#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace visual {
void show_neighbors(size_t id, std::vector<int> &objs) {

  cv::Mat pic = cv::imread("media/" + std::to_string(id) + ".jpg");

  for (auto o : objs) {
    cv::hconcat(cv::imread("media/" + std::to_string(o) + ".jpg"), pic, pic);
  }

  cv::resize(pic, pic, {2000, 1200});
  cv::imshow("Imgs", pic);
  cv::waitKey(0);
}

}; // namespace visual
