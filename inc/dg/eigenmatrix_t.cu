/* test eigenmatrix.h for manualmatrix/FELTOR matrices */

#include <iostream>
#include "manualmatrix.h"
#include "eigenmatrix.h"
#include "elliptic.h"
#include "backend/timer.cuh"

// "difficult physics" for elliptic
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
    std::cout<< "n for random psd and divisor, please:" <<std::endl;
    std::cin>> n_rand >> div;
    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
    std::cout<< "constructing random psd ..." <<std::endl;
    dg::RandPSDmatrix<dg::DVec> spd(n_rand, 1.1);
    dg::EVarbitraryMatrix<dg::DVec> am(n_rand, div);
    double ev_max;
    am( spd, ev_max);
    std::cout << "EV_max: "<< ev_max <<std::endl;
    // now for the elliptic object
    unsigned p, Nx, Ny;
    double eps, jfactor;
    std::cout<< "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" <<std::endl;
    std::cout << "Type p, Nx and Ny and epsilon and jfactor (1), new divisor, too! \n";
    std::cin >> p >> Nx >> Ny;
    std::cin >> eps >> jfactor >> div;
    dg::Grid2d grid( 0, lx, 0, ly, p, Nx, Ny, bcx, bcy);
    dg::DVec chi =  dg::evaluate( pol, grid);
    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
    std::cout << "Create Polarisation object and set chi!\n";
    t.tic();
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> ell( grid, dg::not_normed, dg::centered, jfactor);
    ell.set_chi( chi);
    t.toc();
    std::cout << "Creation of polarisation object took: "<<t.diff()<<"s\n";
    dg::EVarbitraryMatrix<dg::DVec> ae( p*p*Nx*Ny, 100);
    ae( ell, ev_max);
    std::cout << "EV_max: "<< ev_max <<std::endl;
    return 0;
}
