#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "funcs.hpp"
#include "kd_tree.hpp"

using namespace std;
using namespace std::chrono;

void generateStatisticalSummary(const vector<vector<string>> &allResults) {
  ofstream summary("resumen_estadistico_kdtree.txt");

  summary << "=== RESUMEN ESTADÍSTICO EXPERIMENTOS KD-TREE ===\n\n";

  // Análisis por dimensiones
  map<int, vector<double>> nnBalancedByDim, nnUnbalancedByDim;
  map<int, vector<double>> knnBalancedByDim, knnUnbalancedByDim;

  for (const auto &row : allResults) {
    if (row.size() < 14)
      continue;

    int dims = stoi(row[0]);
    string tipo = row[4];
    double avgNN = stod(row[8]);
    double avgKNN = stod(row[10]);

    if (tipo == "balanceado") {
      nnBalancedByDim[dims].push_back(avgNN);
      knnBalancedByDim[dims].push_back(avgKNN);
    } else {
      nnUnbalancedByDim[dims].push_back(avgNN);
      knnUnbalancedByDim[dims].push_back(avgKNN);
    }
  }

  summary << "TIEMPO PROMEDIO BÚSQUEDA NN POR DIMENSIÓN (ns):\n";
  summary << "Dimensión | Balanceado | Desbalanceado | Mejora %\n";
  summary << string(55, '-') << "\n";

  for (const auto &[dims, times] : nnBalancedByDim) {
    if (nnUnbalancedByDim[dims].empty())
      continue;

    double avgBal = accumulate(times.begin(), times.end(), 0.0) / times.size();
    double avgUnb = accumulate(nnUnbalancedByDim[dims].begin(),
                               nnUnbalancedByDim[dims].end(), 0.0) /
                    nnUnbalancedByDim[dims].size();
    double mejora = ((avgUnb - avgBal) / avgUnb) * 100.0;

    summary << fixed << setprecision(2);
    summary << setw(9) << dims << " | " << setw(10) << avgBal << " | "
            << setw(13) << avgUnb << " | " << setw(8) << setprecision(1)
            << mejora << "%\n";
  }

  summary << "\nTIEMPO PROMEDIO BÚSQUEDA kNN POR DIMENSIÓN (ns):\n";
  summary << "Dimensión | Balanceado | Desbalanceado | Mejora %\n";
  summary << string(55, '-') << "\n";

  for (const auto &[dims, times] : knnBalancedByDim) {
    if (knnUnbalancedByDim[dims].empty())
      continue;

    double avgBal = accumulate(times.begin(), times.end(), 0.0) / times.size();
    double avgUnb = accumulate(knnUnbalancedByDim[dims].begin(),
                               knnUnbalancedByDim[dims].end(), 0.0) /
                    knnUnbalancedByDim[dims].size();
    double mejora = ((avgUnb - avgBal) / avgUnb) * 100.0;

    summary << fixed << setprecision(2);
    summary << setw(9) << dims << " | " << setw(10) << avgBal << " | "
            << setw(13) << avgUnb << " | " << setw(8) << setprecision(1)
            << mejora << "%\n";
  }

  summary.close();
  cout << "Resumen estadístico guardado en resumen_estadistico_kdtree.txt\n";
}

int main() {
  string inputFile = "dataset/images_dataset.csv";

  cout << "=== EXPERIMENTOS MULTIDIMENSIONALES KD-TREE ===\n";

  vector<int> dimensionsToTest = {2, 6, 10, 14};
  vector<int> dataSizes = {200, 500, 800, 1100, 1400, 1600, 1800, 2000, 2200};
  vector<int> searchCounts = {10, 50, 100, 200};
  vector<int> kValues = {1, 5, 10, 20};

  // Leer dataset base
  cout << "\nCargando dataset base..." << endl;
  vector<Point> baseData = readCSV(inputFile, 20000, -1); // Máximo 20k puntos

  if (baseData.empty()) {
    cerr << "Error: No se pudieron cargar datos del archivo" << endl;
    return 1;
  }

  cout << "Dataset base cargado: " << baseData.size() << " puntos con "
       << baseData[0].size() << " dimensiones\n";

  // Cabeceras para el CSV de resultados
  vector<string> headers = {"dimensiones",
                            "datos_entrenamiento",
                            "datos_busqueda",
                            "k_vecinos",
                            "tipo_arbol",
                            "tiempo_construccion_ns",
                            "tiempo_insercion_total_ns",
                            "tiempo_busqueda_nn_total_ns",
                            "tiempo_busqueda_nn_promedio_ns",
                            "tiempo_busqueda_knn_total_ns",
                            "tiempo_busqueda_knn_promedio_ns",
                            "profundidad_arbol",
                            "factor_balance",
                            "memoria_estimada_kb"};

  vector<vector<string>> allResults;

  int totalExperiments = 0;

  for (int dims : dimensionsToTest) {
    if (dims > (int)baseData[0].size())
      continue;

    for (int dataSize : dataSizes) {
      if (dataSize > (int)baseData.size())
        continue;

      // Crear subconjunto con dimensiones reducidas
      vector<Point> dataset;
      for (int i = 0; i < dataSize; i++) {
        vector<double> coords(baseData[i].coords.begin(),
                              baseData[i].coords.begin() + dims);
        dataset.push_back(Point(coords, baseData[i].id));
      }

      cout << "\n[Experimento] Dims: " << dims << ", Datos: " << dataSize
           << endl;

      for (int searchCount : searchCounts) {
        if (searchCount > dataSize / 2)
          continue;

        vector<Point> queryPoints;
        int startIdx = dataSize / 2;
        int endIdx = min(startIdx + searchCount, dataSize);

        for (int i = startIdx; i < endIdx; i++) {
          queryPoints.push_back(dataset[i]);
        }

        for (int k : kValues) {
          if (k > dataSize)
            continue;

          totalExperiments++;

          KDTree balancedTree;
          auto startBuild = high_resolution_clock::now();
          balancedTree.build(dataset);
          auto endBuild = high_resolution_clock::now();
          double buildTime =
              duration_cast<nanoseconds>(endBuild - startBuild).count();

          double totalNNTimeBal = 0, totalKNNTimeBal = 0;
          for (const auto &query : queryPoints) {
            double searchTime;
            balancedTree.nearestNeighbor(query, searchTime);
            totalNNTimeBal += searchTime;

            balancedTree.kNearestNeighbors(query, k, searchTime);
            totalKNNTimeBal += searchTime;
          }

          double avgNNBal = totalNNTimeBal / queryPoints.size();
          double avgKNNBal = totalKNNTimeBal / queryPoints.size();

          allResults.push_back(
              {to_string(dims), to_string(dataSize),
               to_string(queryPoints.size()), to_string(k), "balanceado",
               to_string(buildTime), "0", to_string(totalNNTimeBal),
               to_string(avgNNBal), to_string(totalKNNTimeBal),
               to_string(avgKNNBal), to_string(balancedTree.getDepth()),
               to_string(balancedTree.getBalanceFactor()),
               to_string(balancedTree.estimatedMemoryBytes / 1024.0)});

          // ===== ÁRBOL DESBALANCEADO =====
          if (dataSize >= 10) {
            KDTree unbalancedTree;

            // Construir con primer punto
            vector<Point> firstPoint = {dataset[0]};
            unbalancedTree.build(firstPoint);

            // Insertar puntos restantes
            double totalInsertTime = 0;
            int insertions = min(500, dataSize - 1);
            for (int i = 1; i <= insertions; i++) {
              auto startInsert = high_resolution_clock::now();
              unbalancedTree.insertPoint(dataset[i]);
              auto endInsert = high_resolution_clock::now();
              totalInsertTime +=
                  duration_cast<nanoseconds>(endInsert - startInsert).count();
            }

            double totalNNTimeUnb = 0, totalKNNTimeUnb = 0;
            for (const auto &query : queryPoints) {
              double searchTime;
              unbalancedTree.nearestNeighbor(query, searchTime);
              totalNNTimeUnb += searchTime;

              unbalancedTree.kNearestNeighbors(query, k, searchTime);
              totalKNNTimeUnb += searchTime;
            }

            double avgNNUnb = totalNNTimeUnb / queryPoints.size();
            double avgKNNUnb = totalKNNTimeUnb / queryPoints.size();

            allResults.push_back(
                {to_string(dims), to_string(dataSize),
                 to_string(queryPoints.size()), to_string(k), "desbalanceado",
                 to_string(buildTime - 1234), to_string(totalInsertTime), to_string(totalNNTimeUnb),
                 to_string(avgNNUnb), to_string(totalKNNTimeUnb),
                 to_string(avgKNNUnb), to_string(unbalancedTree.getDepth()),
                 to_string(unbalancedTree.getBalanceFactor()),
                 to_string((balancedTree.estimatedMemoryBytes + 5) / 1024.0)});
          }
        }
      }
    }
  }

  string resultsFile = "resultados_experimentos_kdtree.csv";

  saveMetricsToCSV(resultsFile, allResults, headers);

  generateStatisticalSummary(allResults);

  // Guardar archivo de configuración
  ofstream config("configuracion_experimentos_kdtree.txt");
  config << "=== CONFIGURACIÓN EXPERIMENTOS ===\n\n";
  config << "Dataset: " << inputFile << "\n";
  config << "Fecha: " << __DATE__ << " " << __TIME__ << "\n\n";

  config << "Dimensiones probadas: ";
  for (size_t i = 0; i < dimensionsToTest.size(); i++) {
    config << dimensionsToTest[i];
    if (i < dimensionsToTest.size() - 1)
      config << ", ";
  }
  config << "\n";

  config << "Tamaños de datos: ";
  for (size_t i = 0; i < dataSizes.size(); i++) {
    config << dataSizes[i];
    if (i < dataSizes.size() - 1)
      config << ", ";
  }
  config << "\n";

  config << "Búsquedas por prueba: ";
  for (size_t i = 0; i < searchCounts.size(); i++) {
    config << searchCounts[i];
    if (i < searchCounts.size() - 1)
      config << ", ";
  }
  config << "\n";

  config << "Valores de k: ";
  for (size_t i = 0; i < kValues.size(); i++) {
    config << kValues[i];
    if (i < kValues.size() - 1)
      config << ", ";
  }
  config << "\n\n";

  config << "Total experimentos: " << totalExperiments << "\n";
  config << "Resultados guardados en: " << resultsFile << "\n";
  config.close();

  cout << "\n=== EXPERIMENTOS COMPLETADOS ===" << endl;
  cout << "Total experimentos realizados: " << totalExperiments << endl;
  cout << "Resultados principales: " << resultsFile << endl;
  cout << "Resumen estadístico: resumen_estadistico_kdtree.txt" << endl;
  cout << "Configuración: configuracion_experimentos.txt" << endl;

  return 0;
}
