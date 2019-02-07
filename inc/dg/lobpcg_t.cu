#include <stdio.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/print.h>
#include <cusp/gallery/poisson.h>
// include cusp lapack header file
#include <cusp/lapack/lapack.h>
int main()
{
  // create an empty dense matrix structure
  cusp::array2d<float,cusp::host_memory> A;
  // create 2D Poisson problem
  cusp::gallery::poisson5pt(A, 4, 4);
  // compute eigvals and eigvecs
  cusp::array1d<float,cusp::host_memory> eigvals;
  cusp::array2d<float,cusp::host_memory> eigvecs;
  cusp::lapack::sygv(A, A, eigvals, eigvecs);
  // print the eigenvalues
  cusp::print(eigvals);
}
