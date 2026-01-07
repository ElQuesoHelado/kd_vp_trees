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

inline void show_neighbors(
    int ref_id,
    const std::vector<int> &neighbors,
    int offset,
    const std::string &title) {
  constexpr int SCALE = 2;
  constexpr int IMG0 = 125;
  constexpr int IMG = IMG0 * SCALE;
  constexpr int PAD = 20;
  constexpr int COLS = 10;

  auto load = [](int id) {
    cv::Mat img = cv::imread("media/" + std::to_string(id) + ".jpg");
    CV_Assert(!img.empty());
    return img;
  };

  auto border = [](cv::Mat &m, int t, cv::Scalar c) {
    cv::rectangle(m, {0, 0}, {m.cols - 1, m.rows - 1}, c, t);
  };

  cv::Mat ref = load(ref_id);
  cv::resize(ref, ref, {IMG, IMG}, 0, 0, cv::INTER_CUBIC);
  border(ref, 6, {255, 255, 255});

  std::vector<cv::Mat> imgs;
  for (auto &n : neighbors) {
    cv::Mat im = load(n);
    cv::resize(im, im, {IMG, IMG}, 0, 0, cv::INTER_CUBIC);
    border(im, 3, {180, 180, 180});

    imgs.push_back(im);
  }

  int rows = std::ceil(imgs.size() / double(COLS));
  int W = COLS * IMG + (COLS + 1) * PAD;
  int H = IMG + PAD * 3 + rows * (IMG + PAD);

  cv::Mat canvas(H, W, CV_8UC3, {25, 25, 25});

  ref.copyTo(canvas(cv::Rect((W - IMG) / 2, PAD, IMG, IMG)));

  int y0 = IMG + PAD * 2;
  for (size_t i = 0; i < imgs.size(); ++i) {
    int r = i / COLS;
    int c = i % COLS;

    int x = PAD + c * (IMG + PAD);
    int y = y0 + r * (IMG + PAD);

    imgs[i].copyTo(canvas(cv::Rect(x, y, IMG, IMG)));
  }

  std::string win = title + " | id " + std::to_string(ref_id);
  cv::namedWindow(win, cv::WINDOW_NORMAL);
  cv::imshow(win, canvas);

  cv::resizeWindow(win, 1920, 1080);
  cv::moveWindow(win, offset * 40, offset * 40);
}

} // namespace visual
