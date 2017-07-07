#include <iostream>
#include <iomanip>

//#include "backend/xspacelib.cuh"
#include <thrust/device_vector.h>
#include "blas.h"


#include "elliptic.h"
#include "eve.h"
#include "chebyshev.h"
#include "cg.h"
#include "backend/timer.cuh"

//NOTE: IF DEVICE=CPU THEN THE POLARISATION ASSEMBLY IS NOT PARALLEL AS IT IS NOW

//global relative error in L2 norm is O(h^P)
//as a rule of thumb with n=4 the true error is err = 1e-3 * eps as long as eps > 1e3*err

const double lx = M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
//const double eps = 1e-3; //# of pcg iterations increases very much if
// eps << relativer Abstand der exakten Lösung zur Diskretisierung vom Sinus

double initial( double x, double y)
{   return 0.;
}
double amp = 0.9999;
double pol( double x, double y)
{   return 1. + amp*sin(x)*sin(y);    //must be strictly positive
}
//double pol( double x, double y) {return 1.; }
//double pol( double x, double y) {return 1. + sin(x)*sin(y) + x; } //must be strictly positive

double rhs( double x, double y)
{   return 2.*sin(x)*sin(y)*(amp*sin(x)*sin(y)+1)-amp*sin(x)*sin(x)*cos(y)*cos(y)-amp*cos(x)*cos(x)*sin(y)*sin(y);
}
//double rhs( double x, double y) { return 2.*sin( x)*sin(y);}
//double rhs( double x, double y) { return 2.*sin(x)*sin(y)*(sin(x)*sin(y)+1)-sin(x)*sin(x)*cos(y)*cos(y)-cos(x)*cos(x)*sin(y)*sin(y)+(x*sin(x)-cos(x))*sin(y) + x*sin(x)*sin(y);}
double sol(double x, double y)
{   return sin( x)*sin(y);
}
double der(double x, double y)
{   return cos( x)*sin(y);
}


int main()
{   dg::Timer t;
    unsigned n, Nx, Ny;
    double eps, evmaxmul, evminmul;
    double jfactor;
    std::cout << "Type n, Nx and Ny, epsilon, jfactor (1) and ev_max and ev_min multiplier! \n";
    std::cin >> n >> Nx >> Ny; //more N means less iterations for same error
    std::cin >> eps >> jfactor;
    std::cin >> evmaxmul >> evminmul;
    std::cout << "Computation on: "<< n <<" x "<<Nx<<" x "<<Ny<<std::endl;
    //std::cout << "# of 2d cells                 "<< Nx*Ny <<std::endl;
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
    dg::DVec x_inter = dg::evaluate( initial, grid);
    double ev_max;
    std::cout<< "solving with eCG..." <<std::endl;
    {   t.tic();
        dg::eInvert<dg::DVec > invert( x_inter, 500, eps);
        std::cout<< invert( pol, x_inter, b, w2d, chi_inv, v2d, ev_max) << " iterations" <<std::endl;
        t.toc();
        std::cout << "...took " << t.diff() <<std::endl;
        std::cout<< "final EV estimate " << ev_max << std::endl;
        std::cout<< "- - - - - - - - - - - - - - - - - - - -" <<std::endl;
    }
    dg::DVec x_cheb0 = x_inter;//dg::evaluate( initial, grid);
    std::cout<< "solving with Chebychev..." <<std::endl;
    {   t.tic();
      dg::cInvert<dg::DVec > invert( x_cheb0, 1000000, eps);
      //      cheb( A, x_cheb, b, ev_max, ev_min, eps, max_iter)
      std::cout<< invert( pol, x_cheb0, b, eve_max*evmaxmul, eve_max*evminmul, w2d, chi_inv, v2d) << " iterations" <<std::endl;
      t.toc();
      std::cout << "...took " << t.diff() <<std::endl;
      std::cout<< "final EV estimate " << ev_max << std::endl;
      std::cout<< "- - - - - - - - - - - - - - - - - - - -" <<std::endl;
    }




//    std::cout << "Create EVE "<<t.diff()<<"s\n";
//    dg::eInvert<dg::DVec > invert( x, 100, eps);
//    double ev_max;
//    std::cout << eps<<" ";
//    t.tic();
//    std::cout << " "<< invert( pol, x, b, w2d, chi_inv, v2d, ev_max) << std::endl;
//    std::cout << "EV " << ev_max << std::endl;
//    t.toc();
//    std::cout << "Took "<<t.diff()<<"s\n";
}
