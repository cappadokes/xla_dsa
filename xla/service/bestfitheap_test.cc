#include <utility>
#include <fstream>
#include <string>
#include <cassert>
#include <chrono>
#include <map>
#include <iostream>

#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/hlo_value.h"
#include "xla/literal_util.h"
#include "xla/status.h"

int linecnt(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filename << std::endl;
    exit(1);
  }

  int count = 0;
  std::string line;
  while (std::getline(file, line)) {
    count++;
  }

  file.close();

  return count - 1;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <absolute_path_to_csv_file>"
              << std::endl;
    return 1;
  }

  auto algorithm =
      std::make_unique<xla::ConstrainedGlobalDecreasingSizeBestFitHeap>(
          INT64_MAX, 1,
          xla::GlobalDecreasingSizeBestFitHeap<xla::HloValue>::kSpatial);

  std::unique_ptr<xla::HloInstruction> dummy_inst_ =
      xla::HloInstruction::CreateConstant(xla::LiteralUtil::CreateR0(0.0f));

  std::string filepath = argv[1];

  int lines = linecnt(filepath);

  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filepath << std::endl;
    return 1;
  }

  std::string header;
  getline(file, header);

  std::string line;
  xla::HloValue *values[lines];
  int cnt = 0;
  std::map<int, std::pair<int, int>> id_to_interval;
  while (getline(file, line)) {
    std::stringstream ss(line);
    std::string value;

    getline(ss, value, ',');
    auto id = static_cast<int64_t>(stoi(value));

    getline(ss, value, ',');
    auto start = static_cast<int64_t>(stoi(value));

    getline(ss, value, ',');
    auto end = static_cast<int64_t>(stoi(value)) - 1;

    getline(ss, value, ',');
    auto size = static_cast<int64_t>(stoi(value));

    auto val = new xla::HloValue(id, dummy_inst_.get(), {}, false);
    values[cnt] = val;
    algorithm->AllocNew(values[cnt], id, start, end, size);
    cnt++;
    id_to_interval[id] = std::make_pair(start, end);
  }
  file.close();

  assert(cnt == lines);

  auto start = std::chrono::high_resolution_clock::now();
  xla::GlobalDecreasingSizeBestFitHeap<xla::HloValue>::Result result =
      algorithm->FinishNew();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::ofstream timefile("/workspace/xla/xla/service/test-heapsim/times.csv",
                         std::ios::app);
  std::cout << duration.count();

  const char *path = std::getenv("BASE_PATH");
  const char *name = std::getenv("TRACE_NAME");

  if (!(path && name)) {
    std::cerr << "ERROR: One or more environment variables not set!"
              << std::endl;
    return 1;
  }

  std::string path_string = std::string(path);
  std::string filename = std::string(name) + "-out.csv";

  std::string new_path = path_string + "/";
  std::ofstream outfile(new_path + filename, std::ios::trunc);

  if (outfile.is_open()) {
    outfile << "id,lower,upper,size,offset" << std::endl;
    int cnt = 0;
    for (const auto &value : result.heap_results) {
      cnt++;
      if (cnt > 1) {
        std::cerr << "More than 1 heap result found, exiting..." << std::endl;
        exit(1);
      }
      for (const auto &p : value.chunk_map) {
        outfile << p.first->id() << "," << id_to_interval[p.first->id()].first
                << "," << id_to_interval[p.first->id()].second + 1 << ","
                << p.second.size << "," << p.second.offset << std::endl;
      }
    }
    outfile.close();
  } else {
    std::cout << "Could not open file: " << new_path << filename << std::endl;
    exit(1);
  }

  for (int i = 0; i < cnt; ++i) {
    delete values[i];
  }
}