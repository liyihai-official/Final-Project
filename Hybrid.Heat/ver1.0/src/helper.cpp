
#ifndef FINAL_PROJECT_HELPER_CPP_LIYIHAI
#define FINAL_PROJECT_HELPER_CPP_LIYIHAI

#include <helper.hpp>
#include <unordered_map>
#include <iomanip>


namespace final_project
{
  
  Strategy getStrategyfromString(const String& s) {
    const std::unordered_map<std::string, Strategy> String2Strategy = {
        {"PURE_MPI", Strategy::PURE_MPI},
        {"HYBRID_0", Strategy::HYBRID_0},
        {"HYBRID_1", Strategy::HYBRID_1}
    };

    auto it = String2Strategy.find(s);
    if (it != String2Strategy.end()) {
        return it->second;
    } else {
        return Strategy::UNKNOWN;
    }
  }

  std::ostream& operator<<(std::ostream& os, Strategy s)
  {
    switch(s)
    {
      case final_project::Strategy::PURE_MPI   : os << "pure mpi";    break;
      case final_project::Strategy::HYBRID_0   : os << "hybrid 0";    break;
      case final_project::Strategy::HYBRID_1   : os << "hybrid 1";    break;
      default : 
        os << "Unknown strategy\n";
        os.setstate(std::ios_base::failbit);
    }
    return os;
  }


  void version_message(mpi::environment & env)
  {
    if (env.rank() == 0)
      std::cout
        << "final project 6.0"
        << std::endl;
  }

  void helper_message(mpi::environment & env)
  {
    if (env.rank() == 0)
    std::cout 
      << "Usage: mpiexec(mpirun) [OPTION] \n" << "\n"
      << "This is the program of LI Yihai's M.Sc. Final Project. \n" << "\n"
      << "Options:\n" << "\n"
      << "necessary arguments: " << "\n"
      << "\t" << std::left << std::setw(20) << "-s, -S [strategy]" << "\t" 
              << "Specify the parallel evolving strategy using MPI/OpenMP, [strategy] could be " << "\n"
              << "\n"
              << "\t" << std::right << std::setw(34) << "[PURE_MPI]" << "\t" << "Only use Message Passing Interface for communications. "<< "\n"
              << "\t" << std::right << std::setw(34) << "[HYBRID_0]" << "\t" << "First type of MPI/OpenMP hybrid parallel strategy. Using OpenMP for shared memory parallelization." << "\n"
              << "\t" << std::right << std::setw(34) << "[HYBRID_1]" << "\t" << "Second type of MPI/OpenMP hybrid parallel strategy. Using OpenMP for shared memory parallelization." << "\n"
              << "\n"
      << "optional arguments: " << "\n"
      << "\t" << std::left << std::setw(20) << "-h, -H" << "\t" 
              << "Show this help message and exit."
              << "\n"
      << "\t" << std::left << std::setw(20) << "-f, -F [filename]" << "\t"
              << "Target will collect results and save to this file."
              << "\n"
      << "\t" << std::left << std::setw(20) << "-v, -V" << "\t" 
              << "Show the conda version number and exit. \n"
              << "\n"
      << "Build Instruction: \n"
      << "\t" << std::left << std::setw(20) << "Create Makefile " 
      << "\t" << "cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_BUILD_TYPES=[BUILD_TYPE] "
              << "-DNX=[100] -DNY=[100] -DNZ=[50] "
              << "/path/to/projeft/root\n"  << "\n"
      << ""
      << "\t" << std::left << std::setw(20) << "make clean"     << "\t" << "clean all cache files in building processes." << "\n" 
      << "\t" << std::left << std::setw(20) << "make [TARGET]"  << "\t" << "make specified target." << "\n"
      << "\t" << std::left << std::setw(20) << "make -h[--help]" << "\t" << "for showing more make usage." << "\n"
      << "\n"
      << std::endl;
  }
} // namespace final_project





#endif // end define FINAL_PROJECT_HELPER_CPP_LIYIHAI