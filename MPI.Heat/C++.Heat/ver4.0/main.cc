#include "multi_array/base.cpp"


#include <iostream>
#include <iomanip>

int main ()
{

  auto a = final_project::_detail::_multi_array_shape<3>(2, 3, 4);
  final_project::_detail::_multi_array::_array<double,3> b (a);

  b.fill(-1);
  for (std::size_t i = 0; i < b.size(); ++i) b[i] = i+1;

  for (std::size_t i = 0; i < b.size(); ++i) std::cout << b[i] << " ";

  std::cout << std::endl;
  // std::cout << b._shape[0] << " " << b._shape[1] << std::endl;
  std::cout << std::endl;



  for (std::size_t i = 0; i < b._shape[0]; ++i)
  {
    std::cout << "";
    for (std::size_t j = 0; j < b._shape[1]; ++j)
    {
      std::cout << " ";
      for (std::size_t k = 0; k < b._shape[2]; ++k)
      {
        std::cout << " ";
        std::cout << std::fixed << std::setprecision(5) << std::setw(9) << b(i, j, k);
        std::cout << " ";
      }
      std::cout << " ";
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;



  







  return 0;
}