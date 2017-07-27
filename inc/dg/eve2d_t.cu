/* check EVEs capability to estimate the maximum Eigenvalue of Matrices */

#include <iostream>
#include "manualmatrix.h"
#include "eigenmatrix.h"
#include "eve.h"
#include "elliptic.h"
#include "blas.h"

// Imitate "difficult" physical problem
const double lx = M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
double initial( double x, double y)
{   return 0.;
}
double amp = 0.9999;
double pol( double x, double y)
{   return 1. + amp*sin(x)*sin(y);
}
double rhs( double x, double y)
{   return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);
}


int main()
{   // random SPD matrix:
    //   generate Matrix
    int n, div;
    double seed;
    std::cout << "Type n, seed and divisor:" <<std::endl;
    std::cin >> n >> seed >> div;
    dg::RandPSDmatrix<dg::DVec> A(n, seed);
    //   solve with decomposition
    dg::EVarbitraryMatrix<dg::DVec> A_decomp(n, div);
    double ev_decomp;
    A_decomp( A, ev_decomp);
    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
    std::cout<< "max. EV via decomposition: " << ev_decomp <<std::endl;
    //   solve with EVE
    //   ... needs dummy x and b as EVE works on solving Ax = b
    dg::DVec x(n, 0.0), b(n, 1.0);
    dg::EVE<dg::DVec> A_eve( x, n);
    double ev_eve;
    unsigned niter = A_eve( A, x, b, ev_eve);
    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
    std::cout<< "max. EV via EVEs guess: " << ev_eve << " after "<< niter << " iterations"<<std::endl;
    std::cout<< "+ + + + + + + + + + + + + + +" <<std::endl;
    std::cout<< "decomp-EVE: " << ev_decomp - ev_eve <<std::endl;
    std::cout<< "+ + + + + + + + + + + + + + +" <<std::endl;

    // for an elliptic object
    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
    unsigned Nx, Ny, p;
    double jfactor;
    std::cout << "Type p, Nx and Ny and jfactor (1) as well as divisor! \n";
    std::cin >> p >> Nx >> Ny; //more N means less iterations for same error
    std::cin >> jfactor >> div;
    dg::Grid2d grid( 0, lx, 0, ly, p, Nx, Ny, bcx, bcy);
    dg::DVec w2d = dg::create::weights( grid);
    dg::DVec v2d = dg::create::inv_weights( grid);
    dg::DVec one = dg::evaluate( dg::one, grid);
    b = dg::evaluate( rhs, grid);
    dg::DVec chi =  dg::evaluate( pol, grid);
    dg::DVec chi_inv(chi);
    dg::blas1::transform( chi, chi_inv, dg::INVERT<double>());
    dg::blas1::pointwiseDot( chi_inv, v2d, chi_inv);
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> ell( grid, dg::not_normed, dg::centered, jfactor);
    ell.set_chi( chi);
    std::cout << "Created Polarisation object and set chi!\n";
    //   solve with decomposition
    dg::EVarbitraryMatrix<dg::DVec> ell_decomp(p*p*Nx*Ny, div);
    ell_decomp( ell, ev_decomp);
    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
    std::cout<< "max. EV via decomposition: " << ev_decomp <<std::endl;
    //   solve with EVE
    x = dg::evaluate( initial, grid);
    dg::EVE<dg::DVec> ell_eve( x, p*p*Nx*Ny);
    double ev_max;
    niter = ell_eve( ell, x, b, ev_max);
    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
    std::cout<< "max. EV via EVEs guess: " << ev_max << " after "<< niter << " iterations"<<std::endl;
    //    does it work with eInvert (tested as mere wrapper for EVE)
    b = dg::evaluate( rhs, grid);
    x = dg::evaluate( initial, grid);
    dg::eInvert<dg::DVec> invert( x, p*p*Nx*Ny, 1e-6, 1, false);
    niter = invert( ell, x, b, w2d, chi_inv, v2d, ev_eve);
    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
    std::cout<< "max. EV via eInvert guess: " << ev_eve << " after "<< niter << " iterations"<<std::endl;
    return 0;
}
