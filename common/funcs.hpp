#pragma once

#include "point.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>

inline vector<Point> readCSV(const string &filename, int maxRows = -1,
                             int numFeatures = -1) {
  ifstream file(filename);
  vector<Point> points;
  string line;
  int rowCount = 0;

  if (!file.is_open()) {
    cerr << "Error: No se pudo abrir el archivo " << filename << endl;
    return points;
  }

  while (getline(file, line) && (maxRows == -1 || rowCount < maxRows)) {
    stringstream ss(line);
    string value;

    if (!getline(ss, value, ','))
      continue;

    int id;
    try {
      id = stoi(value);
    } catch (...) {
      // cerr << "Error al leer ID: " << value << endl;
      continue;
    }

    vector<double> coords;
    int featureCount = 0;

    while (getline(ss, value, ',')) {
      if (numFeatures != -1 && featureCount >= numFeatures)
        break;

      try {
        coords.push_back(stod(value));
        featureCount++;
      } catch (...) {
        cerr << "Error al convertir valor: " << value << endl;
        coords.push_back(0.0);
        featureCount++;
      }
    }

    if (!coords.empty()) {
      points.emplace_back(coords, id);
      rowCount++;
    }
  }

  file.close();
  return points;
}

inline void saveMetricsToCSV(const string &filename,
                             const vector<vector<string>> &metrics,
                             const vector<string> &headers) {
  ofstream file(filename);

  if (!file.is_open()) {
    cerr << "Error: No se pudo crear el archivo " << filename << endl;
    return;
  }

  for (size_t i = 0; i < headers.size(); ++i) {
    file << headers[i];
    if (i < headers.size() - 1)
      file << ",";
  }
  file << endl;

  for (const auto &row : metrics) {
    for (size_t i = 0; i < row.size(); ++i) {
      file << row[i];
      if (i < row.size() - 1)
        file << ",";
    }
    file << endl;
  }

  file.close();
  cout << "Métricas guardadas en " << filename << endl;
}
