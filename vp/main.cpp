#include "point.hpp"
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "funcs.hpp"
#include "vp_tree.hpp"

using namespace std;
using namespace std::chrono;

void generateVPStatisticalSummary(const vector<vector<string>> &results,
                                  const vector<double> &allPruningRates,
                                  const vector<double> &allBuildTimes,
                                  const vector<int> &dimensionsToTest,
                                  const vector<int> &dataSizes) {
  ofstream summary("resumen_estadistico_vptree.txt");

  summary << "=== RESUMEN ESTADÍSTICO VP-TREE ===\n\n";

  // Estadísticas generales
  if (!allPruningRates.empty()) {
    double avgPruning = accumulate(allPruningRates.begin(), allPruningRates.end(), 0.0) / allPruningRates.size();
    double maxPruning = *max_element(allPruningRates.begin(), allPruningRates.end());
    double minPruning = *min_element(allPruningRates.begin(), allPruningRates.end());

    summary << fixed << setprecision(3);
    summary << "ESTADÍSTICAS GENERALES:\n";
    summary << "=======================\n";
    summary << "Tasa de poda promedio: " << avgPruning << endl;
    summary << "Tasa de poda máxima:   " << maxPruning << endl;
    summary << "Tasa de poda mínima:   " << minPruning << endl;
    summary << "Experimentos totales:  " << allPruningRates.size() << endl;
  }

  if (!allBuildTimes.empty()) {
    double avgBuild = accumulate(allBuildTimes.begin(), allBuildTimes.end(), 0.0) / allBuildTimes.size();
    double maxBuild = *max_element(allBuildTimes.begin(), allBuildTimes.end());
    double minBuild = *min_element(allBuildTimes.begin(), allBuildTimes.end());

    summary << "\nTiempo construcción promedio: " << avgBuild << " ns" << endl;
    summary << "Tiempo construcción máximo:   " << maxBuild << " ns" << endl;
    summary << "Tiempo construcción mínimo:   " << minBuild << " ns" << endl;
  }

  summary << "\nANÁLISIS POR DIMENSIÓN:\n";
  summary << "=======================\n";

  map<int, vector<double>> pruningByDim;
  map<int, vector<double>> searchTimeByDim;
  map<int, vector<double>> buildTimeByDim;
  map<int, vector<double>> radiusByDim;

  for (const auto &row : results) {
    if (row.size() < 15)
      continue;

    int dims = stoi(row[0]);            // dimensiones
    double buildTime = stod(row[4]);    // tiempo_construccion_ms
    double searchTime = stod(row[6]);   // tiempo_busqueda_nn_promedio_ms
    double pruningRate = stod(row[10]); // tasa_poda_promedio
    double radius = stod(row[13]);      // radio_promedio_particion

    pruningByDim[dims].push_back(pruningRate);
    searchTimeByDim[dims].push_back(searchTime);
    buildTimeByDim[dims].push_back(buildTime);
    radiusByDim[dims].push_back(radius);
  }

  for (int dim : dimensionsToTest) {
    if (pruningByDim.find(dim) != pruningByDim.end() && !pruningByDim[dim].empty()) {
      double avgPruning = accumulate(pruningByDim[dim].begin(), pruningByDim[dim].end(), 0.0) / pruningByDim[dim].size();
      double avgSearch = accumulate(searchTimeByDim[dim].begin(), searchTimeByDim[dim].end(), 0.0) / searchTimeByDim[dim].size();
      double avgBuild = accumulate(buildTimeByDim[dim].begin(), buildTimeByDim[dim].end(), 0.0) / buildTimeByDim[dim].size();
      double avgRadius = accumulate(radiusByDim[dim].begin(), radiusByDim[dim].end(), 0.0) / radiusByDim[dim].size();

      summary << "\nDIMENSIÓN " << dim << ":\n";
      summary << "  Tasa poda promedio:    " << avgPruning << endl;
      summary << "  Tiempo búsqueda NN:    " << avgSearch << " ns" << endl;
      summary << "  Tiempo construcción:   " << avgBuild << " ns" << endl;
      summary << "  Radio partición:       " << avgRadius << endl;
      summary << "  Muestras:              " << pruningByDim[dim].size() << endl;
    }
  }

  summary << "\n\nANÁLISIS POR TAMAÑO DE DATOS:\n";
  summary << "==============================\n";

  map<int, vector<double>> pruningBySize;
  map<int, vector<double>> depthBySize;

  for (const auto &row : results) {
    if (row.size() < 15)
      continue;

    int size = stoi(row[1]);            // datos_entrenamiento
    double pruningRate = stod(row[10]); // tasa_poda_promedio
    double depth = stod(row[9]);        // profundidad_arbol

    pruningBySize[size].push_back(pruningRate);
    depthBySize[size].push_back(depth);
  }

  for (int size : dataSizes) {
    if (pruningBySize.find(size) != pruningBySize.end() && !pruningBySize[size].empty()) {
      double avgPruning = accumulate(pruningBySize[size].begin(), pruningBySize[size].end(), 0.0) / pruningBySize[size].size();
      double avgDepth = accumulate(depthBySize[size].begin(), depthBySize[size].end(), 0.0) / depthBySize[size].size();

      summary << "Tamaño " << setw(4) << size
              << " | Poda: " << setw(6) << avgPruning
              << " | Profundidad: " << setw(5) << avgDepth
              << " | Muestras: " << pruningBySize[size].size() << endl;
    }
  }

  // Correlaciones clave
  summary << "\n\nOBSERVACIONES CLAVE:\n";
  summary << "====================\n";

  // Calcular correlación dimensión vs tasa de poda
  vector<double> dimValues, pruneValues;
  for (const auto &[dim, prunes] : pruningByDim) {
    if (!prunes.empty()) {
      double avgPrune = accumulate(prunes.begin(), prunes.end(), 0.0) / prunes.size();
      dimValues.push_back(dim);
      pruneValues.push_back(avgPrune);
    }
  }

  if (dimValues.size() > 1) {
    // Calcular tendencia simple
    summary << "1. Tendencia Dimensión vs Poda: ";
    if (pruneValues.back() > pruneValues.front()) {
      summary << "La poda MEJORA con dimensiones más altas\n";
    } else if (pruneValues.back() < pruneValues.front()) {
      summary << "La poda EMPEORA con dimensiones más altas\n";
    } else {
      summary << "La poda se mantiene estable con dimensiones\n";
    }
  }

  // Efecto del tamaño en profundidad
  vector<double> sizeValues, depthValues;
  for (const auto &[size, depths] : depthBySize) {
    if (!depths.empty()) {
      double avgDepth = accumulate(depths.begin(), depths.end(), 0.0) / depths.size();
      sizeValues.push_back(size);
      depthValues.push_back(avgDepth);
    }
  }

  if (sizeValues.size() > 1) {
    summary << "2. Crecimiento de profundidad: O(log n) aproximado\n";
    summary << "3. Radio de partición típico: "
            << (radiusByDim.empty() ? "N/A" : to_string(radiusByDim.begin()->second[0])) << endl;
  }

  summary.close();
}

int main() {
  string inputFile = "dataset/images_dataset.csv";

  cout << "=== EXPERIMENTOS MULTIDIMENSIONALES VP-TREE ===\n";

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
                            "tiempo_construccion_ns",
                            "tiempo_busqueda_nn_total_ns",
                            "tiempo_busqueda_nn_promedio_ns",
                            "tiempo_busqueda_knn_total_ns",
                            "tiempo_busqueda_knn_promedio_ns",
                            "profundidad_arbol",
                            "tasa_poda_promedio",
                            "llamadas_distancia_total",
                            "nodos_visitados_promedio",
                            "radio_promedio_particion",
                            "memoria_estimada_kb"};

  vector<vector<string>> allResults;

  int totalExperiments = 0;

  // Para análisis estadístico de aleatoriedad
  vector<double> allPruningRates;
  vector<double> allBuildTimes;

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

          VP_tree vpTree(dataset);

          auto startBuild = high_resolution_clock::now();
          vpTree.build();
          auto endBuild = high_resolution_clock::now();
          double buildTime =
              duration_cast<nanoseconds>
(endBuild - startBuild).count();

          allBuildTimes.push_back(buildTime);

          vpTree.reset_search_metrics();

          double totalNNTime = 0, totalKNNTime = 0;
          double totalPruningRate = 0;
          double totalVisitedNodes = 0;

          for (const auto &query : queryPoints) {
            double searchTime;

            // NN
            vpTree.reset_search_metrics();

            auto start = high_resolution_clock::now();
            vpTree.nn(query.id);
            auto end = high_resolution_clock::now();
            searchTime = duration_cast<nanoseconds>
(end - start).count();

            totalNNTime += searchTime;
            double pruningRateNN = vpTree.get_last_prunning_rate();
            double visitedNodesNN = vpTree.get_last_visited_nodes();

            // k-NN
            vpTree.reset_search_metrics();

            start = high_resolution_clock::now();
            vpTree.knn(query.id, k);
            end = high_resolution_clock::now();
            searchTime = duration_cast<nanoseconds>
(end - start).count();

            totalKNNTime += searchTime;

            totalPruningRate += (pruningRateNN + vpTree.get_last_prunning_rate()) / 2.0;
            totalVisitedNodes += (visitedNodesNN + vpTree.get_last_visited_nodes()) / 2.0;
          }

          double avgNN = totalNNTime / queryPoints.size();
          double avgKNN = totalKNNTime / queryPoints.size();
          double avgPruningRate = totalPruningRate / queryPoints.size();
          double avgVisitedNodes = totalVisitedNodes / queryPoints.size();

          allPruningRates.push_back(avgPruningRate);

          // Estadisticas globales
          double avgPartitionRadius = vpTree.get_average_partition_radius();
          long totalDistanceCalls = vpTree.get_total_distance_calls();

          // Guardar resultados
          allResults.push_back(
              {to_string(dims),
               to_string(dataSize),
               to_string(queryPoints.size()),
               to_string(k),
               to_string(buildTime),
               to_string(totalNNTime),
               to_string(avgNN),
               to_string(totalKNNTime),
               to_string(avgKNN),
               to_string(vpTree.get_depth()),
               to_string(avgPruningRate),
               to_string(totalDistanceCalls),
               to_string(avgVisitedNodes),
               to_string(avgPartitionRadius),
               to_string(vpTree.estimatedMemoryBytes / 1024.0)});

          cout << "  [VP-Tree] Dims: " << dims
               << ", Tamaño: " << dataSize
               << ", Poda: " << avgPruningRate
               << ", Distancias: " << totalDistanceCalls
               << ", Radio: " << avgPartitionRadius << endl;
        }
      }
    }
  }

  string resultsFile = "resultados_experimentos_vptree.csv";

  saveMetricsToCSV(resultsFile, allResults, headers);

  // Análisis estadístico para VP-Tree
  generateVPStatisticalSummary(allResults, allPruningRates, allBuildTimes, dimensionsToTest, dataSizes);

  ofstream config("configuracion_experimentos_vptree.txt");
  config << "=== CONFIGURACIÓN EXPERIMENTOS VP-TREE ===\n\n";
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

  config << "PARÁMETROS VP-TREE:\n";
  config << "  - Selección VP: Aleatoria\n";
  config << "  - Métrica distancia: Euclidiana\n";
  config << "  - Construcción: Estática (no incremental)\n\n";

  config << "Total experimentos: " << totalExperiments << "\n";
  config << "Resultados guardados en: " << resultsFile << "\n";

  config.close();

  cout << "\n=== EXPERIMENTOS VP-TREE COMPLETADOS ===" << endl;
  cout << "Total experimentos realizados: " << totalExperiments << endl;
  cout << "Resultados: " << resultsFile << endl;
  cout << "Resumen estadístico: resumen_estadistico_vptree.txt" << endl;
  cout << "Configuración: configuracion_experimentos_vptree.txt" << endl;

  return 0;
}
