#include <iostream>
#include "manualmatrix.h"
#include "analysematrix.h"
#include "elliptic.h"
#include "backend/timer.cuh"

/* test analysematrix.h for available classes */

const double lx = M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
double initial( double x, double y)
{   return 0.;
}
double amp = 0.9999;
double pol( double x, double y)
{   return 1. + amp*sin(x)*sin(y);    //must be strictly positive
}
double rhs( double x, double y)
{   return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);
}

int main()
{   dg::Timer t;
    // for some random symmetric positive matrix
    int n_rand, div;
//    std::cout<< "n for random psd and divisor, please:" <<std::endl;
//    std::cin>> n_rand >> div;
//    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
//    std::cout<< "constructing random psd ..." <<std::endl;
//    dg::RandPSDmatrix<dg::DVec> spd(n_rand, 1.1);
//    dg::AnalysisMatrix<dg::DVec> am(n_rand, n_rand);
    double ev_max;
//    am( spd, div, ev_max);
    // now for the elliptic object
    unsigned p, Nx, Ny;
    double eps, jfactor;
    std::cout << "Type p, Nx and Ny and epsilon and jfactor (1), new divisor, too! \n";
    std::cin >> p >> Nx >> Ny;
    std::cin >> eps >> jfactor >> div;
    dg::Grid2d grid( 0, lx, 0, ly, p, Nx, Ny, bcx, bcy);
    dg::DVec chi =  dg::evaluate( pol, grid);
    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
    std::cout << "Create Polarisation object and set chi!\n";
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> ell( grid, dg::not_normed, dg::centered, jfactor);
    ell.set_chi( chi);
    dg::AnalysisMatrix<dg::DVec> ae(p*p*Nx*Ny, p*p*Nx*Ny);
    ae( ell, div, ev_max);
    return 0;
}
