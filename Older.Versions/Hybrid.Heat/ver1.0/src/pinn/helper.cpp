#ifndef FINAL_PROJECT_PINN_HELPER_CPP_LIYIHAI
#define FINAL_PROJECT_PINN_HELPER_CPP_LIYIHAI

#include <pinn/helper.hpp>
#include <unordered_map>
#include <iomanip>

namespace final_project 
{
  namespace PINN {
    Dimension getDimensionfromString(const String & s)
    {
      const std::unordered_map<String, Dimension> String2Dimension = {
        {"2D", Dimension::PINN_2D},
        {"3D", Dimension::PINN_3D}
      };

      auto it = String2Dimension.find(s);
      if (it != String2Dimension.end())
      {
        return it->second;
      } else {
        return Dimension::UNKNOWN;
      }
    }

    std::ostream & operator<<(std::ostream & os, Dimension D)
    {
      switch(D)
      {
        case final_project::PINN::Dimension::PINN_2D : os << "2D PINN"; break;
        case final_project::PINN::Dimension::PINN_3D : os << "3D PINN"; break;
        default:
          os << "Unknown dimension of PINN\n";
          os.setstate(std::ios_base::failbit);
      }
      return os;
    }

    void helper_message()
    {
      std::cout 
        << "Usage: <path/to/executable> \n" << "\n"
        << "This is the program of LI Yihai's M.Sc. Final Project. \n" << "\n"
        << "Options: \n" << "\n"
        << "necessary arguments: " << "\n"
        << "\t" << std::left << std::setw(20) << "-d, -D [Dimension]" << "\t"
        << std::endl;
    }
  }
}



#endif // end define FINAL_PROJECT_PINN_HELPER_CPP_LIYIHAI