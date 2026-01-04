#pragma once

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

namespace visual {
inline void show_neighbors(size_t id, std::vector<int> &objs) {
  cv::Mat pic = cv::imread("media/" + std::to_string(id) + ".jpg");

  for (auto o : objs) {
    cv::hconcat(cv::imread("media/" + std::to_string(o) + ".jpg"), pic, pic);
  }

  cv::resize(pic, pic, {2000, 400});

  std::string window_name = "Vecinos de " + std::to_string(id);

  cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

  cv::resizeWindow(window_name, pic.cols, pic.rows);

  cv::imshow(window_name, pic);

  cv::resizeWindow(window_name, pic.cols, pic.rows);

  std::cout << "Presiona cualquier tecla para continuar..." << std::endl;
  cv::waitKey(0);
  cv::destroyWindow(window_name);
}
}; // namespace visual
