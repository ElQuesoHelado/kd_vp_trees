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

  cv::resize(pic, pic, {2000, 600});

  std::string window_name = "Vecinos de " + std::to_string(id);

  // Crear ventana con flag para mantener tamaño
  cv::namedWindow(window_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

  // Forzar tamaño de ventana ANTES de mostrar
  cv::resizeWindow(window_name, pic.cols, pic.rows);

  // Ahora mostrar la imagen
  cv::imshow(window_name, pic);

  // También forzar después por si acaso
  cv::resizeWindow(window_name, pic.cols, pic.rows);

  std::cout << "Presiona cualquier tecla para continuar..." << std::endl;
  cv::waitKey(0);
  cv::destroyWindow(window_name);
}
}; // namespace visual
