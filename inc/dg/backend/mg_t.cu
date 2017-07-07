#include <iostream>

#include <cusp/print.h>
#include "xspacelib.cuh"
#include "interpolation.cuh"
#include "../blas.h"
#include "evaluation.cuh"

#include <math.h>

double function( double x, double y)
{   return sin(x)*cos(y/4.);
}
typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;

int main()
{   double x0 = 0., x1 = 2.*M_PI;
    int p = 3, n = 4;

    dg::Grid2d grid_2h(x0, x1, x0, x1, p, n/2, n/2);
    dg::DVec vec_2h = dg::evaluate( function, grid_2h);
    for( unsigned i=0; i<vec_2h.size(); ++i)
    {   std::cout<< vec_2h[i] << std::endl;
    }
    std::cout << "----------" <<std::endl;
    dg::Grid2d grid_1h(x0, x1, x0, x1, p, n, n);
    Matrix C2F = dg::create::interpolation( grid_1h, grid_2h);
    dg::DVec vec_1h = dg::evaluate( dg::one, grid_1h);
    dg::blas2::symv( C2F, vec_2h, vec_1h);
//    for( unsigned i=0; i<vec_1h.size(); ++i)
//    {   std::cout<< vec_1h[i] << std::endl;
//    }
//    std::cout << "----------" <<std::endl;
    Matrix F2C = dg::create::interpolation( grid_2h, grid_1h);
    dg::blas2::symv( F2C, vec_1h, vec_2h);
    for( unsigned i=0; i<vec_2h.size(); ++i)
    {   std::cout<< vec_2h[i] << std::endl;
    }


//    dg::Grid1d grid_2h(x0, x1, p, n/2);
//    dg::DVec vec_2h = dg::evaluate( function, grid_2h);
//    for( unsigned i=0; i<vec_2h.size(); ++i)
//      {   std::cout<< vec_2h[i] << std::endl;
//      }
//    std::cout << "----------" <<std::endl;
//    dg::Grid1d grid_1h(x0, x1, p, n);
//    Matrix C2F = dg::create::interpolation( grid_1h, grid_2h);
//    dg::DVec vec_1h = dg::evaluate( dg::one, grid_1h);
//    dg::blas2::symv( C2F, vec_2h, vec_1h);
//    for( unsigned i=0; i<vec_1h.size(); ++i)
//    {   std::cout<< vec_1h[i] << std::endl;
//    }
//    std::cout << "----------" <<std::endl;
//    Matrix F2C = dg::create::interpolation( grid_2h, grid_1h);
//    dg::blas2::symv( F2C, vec_1h, vec_2h);
//    for( unsigned i=0; i<vec_2h.size(); ++i)
//      {   std::cout<< vec_2h[i] << std::endl;
//      }
    return 0;
}
