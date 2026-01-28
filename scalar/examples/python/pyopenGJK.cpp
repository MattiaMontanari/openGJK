#include "openGJK/openGJK.h"
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;

PYBIND11_MODULE(opengjkc, m) {
  m.def("gjk",
        [](Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> &arr1,
           Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> &arr2)
            -> double {
          gkSimplex s;
          gkPolytope bd1;
          gkPolytope bd2;
          bd1.numpoints = arr1.rows();
          std::vector<double *> arr1_rows(arr1.rows());
          for (int i = 0; i < arr1.rows(); ++i)
            arr1_rows[i] = arr1.row(i).data();
          bd1.coord = arr1_rows.data();

          bd2.numpoints = arr2.rows();
          std::vector<double *> arr2_rows(arr2.rows());
          for (int i = 0; i < arr2.rows(); ++i)
            arr2_rows[i] = arr2.row(i).data();
          bd2.coord = arr2_rows.data();

          double a = compute_minimum_distance(bd1, bd2, &s);

          return a;
        });
}
