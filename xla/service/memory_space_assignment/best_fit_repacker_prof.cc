#include "xla/service/memory_space_assignment/best_fit_repacker.h"

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/service/heap_simulator/allocation_block.h"
#include "xla/service/heap_simulator/heap_simulator.h"
#include "xla/service/memory_space_assignment/repacking.h"
#include "xla/statusor.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>

namespace xla {

class MemorySpaceAssignmentBestFitRepackerTest {
 public:
  MemorySpaceAssignmentBestFitRepackerTest()
      : repacker_(INT64_MAX, 1, SliceTimePermutationIterator::Ty::kAll,
                  options_) {}

  AllocationBlock *MakeAllocationBlock(int64_t start_time, int64_t end_time,
                                       int64_t size,
                                       int64_t initial_offset = -1) {
    allocation_blocks_.push_back(
        {start_time, end_time, size, -1, initial_offset,
         static_cast<int64_t>(allocation_blocks_.size())});
    AllocationBlock *block = &allocation_blocks_.back();
    block->next_colocated = block;
    return block;
  }

  std::list<AllocationBlock> allocation_blocks_;
  memory_space_assignment::MemorySpaceAssignmentBestFitRepacker::
      BestFitRepackOptions options_{/*validate=*/true,
                                    /*buffer_interval_compare=*/nullptr};
  memory_space_assignment::MemorySpaceAssignmentBestFitRepacker repacker_;
};

}  // namespace xla

std::vector<xla::AllocationBlock *> fillAllocations(
    const std::string &filepath,
    xla::MemorySpaceAssignmentBestFitRepackerTest &RepackerTest) {
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Error: Unable to open file " << filepath << std::endl;
    exit(1);
  }

  std::string header;
  getline(file, header);

  std::string line;
  std::vector<xla::AllocationBlock *> allocation_blocks;
  while (getline(file, line)) {
    std::stringstream ss(line);
    std::string value;

    // skip id
    getline(ss, value, ',');

    getline(ss, value, ',');
    auto start = static_cast<int64_t>(stoi(value));

    getline(ss, value, ',');
    auto end = static_cast<int64_t>(stoi(value)) - 1;

    getline(ss, value, ',');
    auto size = static_cast<int64_t>(stoi(value));

    allocation_blocks.push_back(
        RepackerTest.MakeAllocationBlock(start, end, size));
  }
  file.close();
  return allocation_blocks;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <absolute_path_to_csv_file>"
              << std::endl;
    return 1;
  }

  std::string filepath = argv[1];
  xla::MemorySpaceAssignmentBestFitRepackerTest RepackerTest =
      xla::MemorySpaceAssignmentBestFitRepackerTest();
  std::vector<xla::AllocationBlock *> allocation_blocks =
      fillAllocations(filepath, RepackerTest);

  const char *path = std::getenv("BASE_PATH");
  const char *name = std::getenv("TRACE_NAME");

  if (!(path && name)) {
    std::cerr << "ERROR: One or more environment variables not set!"
              << std::endl;
    return 1;
  }

  auto start = std::chrono::high_resolution_clock::now();
  absl::StatusOr<bool> success =
      RepackerTest.repacker_.Repack(absl::MakeSpan(allocation_blocks));
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << duration.count();

  std::string path_string = std::string(path);
  std::string filename = std::string(name) + "-out.csv";

  std::string new_path = path_string + "/" + "csv-out" + "/";
  std::ofstream outfile(new_path + filename, std::ios::trunc);

  if (outfile.is_open()) {
    outfile << "id,lower,upper,size,offset" << std::endl;
    for (auto AllocationBlockPtr : RepackerTest.allocation_blocks_) {
      outfile << AllocationBlockPtr.id << ","
              << AllocationBlockPtr.inclusive_start_time << ","
              << AllocationBlockPtr.end_time + 1 << ","
              << AllocationBlockPtr.size << "," << AllocationBlockPtr.offset
              << std::endl;
    }
    outfile.close();

  } else {
    std::cout << "Could not open file: " << new_path << filename << std::endl;
    exit(1);
  }
}
