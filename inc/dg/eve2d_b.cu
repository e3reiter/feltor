#include <iostream>
#include <iomanip>

#include "blas.h"
#include "elliptic.h"
#include "eve.h"
#include "chebyshev.h"
#include "cg.h"
#include "backend/timer.cuh"
#include "manualmatrix.h"


/* Test maximum eigenvalue estimations by eve.h:
   - for elliptic via eInvert
   - for random matrix from manualmatrix.h       */

// Generate analytical test for elliptic
const double lx = M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;

double initial( double x, double y)
{   return 0.;
}
double amp = 0.9999; // the closer to one the worse for the solver
double pol( double x, double y)
{   return 1. + amp*sin(x)*sin(y);    //must be strictly positive
}
double rhs( double x, double y)
{   return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);
}
double sol(double x, double y)
{   return sin( x)*sin(y);
}
double der(double x, double y)
{   return cos( x)*sin(y);
}


int main()
{   dg::Timer t;
  /* elliptic example */
    unsigned n, Nx, Ny;
    double eps, jfactor;
    std::cout << "Type n, Nx and Ny, epsilon, jfactor (1)! \n";
    std::cin >> n >> Nx >> Ny; //more N means less iterations for same error
    std::cin >> eps >> jfactor;
    std::cout << "Computation on: "<< n <<" x "<<Nx<<" x "<<Ny<<std::endl;
    dg::Grid2d grid( 0, lx, 0, ly, n, Nx, Ny, bcx, bcy);
    dg::DVec w2d = dg::create::weights( grid);
    dg::DVec v2d = dg::create::inv_weights( grid);
    dg::DVec one = dg::evaluate( dg::one, grid);
    //create functions A(chi) x = b
    dg::DVec b =    dg::evaluate( rhs, grid);
    dg::DVec chi =  dg::evaluate( pol, grid);
    dg::DVec chi_inv(chi);
    dg::blas1::transform( chi, chi_inv, dg::INVERT<double>());
    dg::blas1::pointwiseDot( chi_inv, v2d, chi_inv);
    std::cout << "Create Polarisation object and set chi!\n";
    t.tic();
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> pol( grid, dg::not_normed, dg::centered, jfactor);
    pol.set_chi( chi);
    t.toc();
    std::cout << "Creation of polarisation object took: "<<t.diff()<<"s\n";

    //solve with pCG
    dg::DVec x_pcg = dg::evaluate( initial, grid);
    std::cout<< "solving with PC..." <<std::endl;
    {   t.tic();
        dg::Invert<dg::DVec > invert( x_pcg, n*n*Nx*Ny, eps);
        std::cout<< invert( pol, x_pcg, b, w2d, chi_inv, v2d) << " iterations"<<std::endl;
        t.toc();
        std::cout<< "...took " << t.diff() <<std::endl;
        std::cout<< "- - - - - - - - - - - - - - - - - - - -" <<std::endl;
    }
    dg::DVec x_ecg = dg::evaluate( initial, grid);
    double eve_max;
    std::cout<< "solving with eCG..." <<std::endl;
    {   t.tic();
        dg::eInvert<dg::DVec > invert( x_ecg, n*n*Nx*Ny, eps);
        std::cout<< invert( pol, x_ecg, b, w2d, chi_inv, v2d, eve_max) << " iterations" <<std::endl;
        t.toc();
        std::cout << "...took " << t.diff() <<std::endl;
        std::cout<< "final EV estimate " << eve_max << std::endl;
        std::cout<< "- - - - - - - - - - - - - - - - - - - -" <<std::endl;
    }
    /* random spd matrix */
    unsigned N;
    std::cout << "Type N of SPD Matrix \n";
    std::cin >> N;
    dg::RandPSDmatrix<dg::DVec> spd(N, 1.1);
    std::cout<< "Eigen max. EV " << spd.get_maxev() << std::endl;
    std::cout<< "- - - - - - - - - - - - - - - - - - - -" <<std::endl;
    dg::DVec alpha(N), beta(N);
    for( uint i=0; i<N; ++i)
      { alpha[i] = 0.0;
        beta[i] = 1.0;
      }
    dg::EVE<dg::DVec> eve(alpha, N*N);
    eve(spd, alpha, beta, eve_max);
    std::cout<< "EVE max. EV " << eve_max << std::endl;
    std::cout<< "- - - - - - - - - - - - - - - - - - - -" <<std::endl;
}
