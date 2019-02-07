/* check Chebyshev as a linear solver */

#include <iostream>
#include "manualmatrix.h"
#include "eigenmatrix.h"
#include "eve.h"
#include "elliptic.h"
#include "chebyshev.h"
#include "cg.h"

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

template< class Vector>
void printvector( Vector& v)
{ for( uint i=0; i<v.size(); ++i)
    {   std::cout<<v[i]<<std::endl;
    }
  std::cout<<"- - - - - - - -"<<std::endl;
}


int main()
{   unsigned p, Nx, Ny;
    double eps;
    double jfactor;
    std::cout << "Type p, Nx and Ny and epsilon and jfactor (1.0)! \n";
    std::cin >> p >> Nx >> Ny; //more N means less iterations for same error
    std::cin >> eps >> jfactor;
    std::cout << "Computation on: "<< p <<" x "<<Nx<<" x "<<Ny<<std::endl;
    //std::cout << "# of 2d cells                 "<< Nx*Ny <<std::endl;
    dg::Grid2d grid( 0, lx, 0, ly, p, Nx, Ny, bcx, bcy);
    dg::DVec w2d = dg::create::weights( grid);
    dg::DVec v2d = dg::create::inv_weights( grid);
    dg::DVec one = dg::evaluate( dg::one, grid);
    //create functions A(chi) x = b
    dg::DVec x =    dg::evaluate( initial, grid);
    dg::DVec b =    dg::evaluate( rhs, grid);
    dg::DVec chi =  dg::evaluate( pol, grid);
    dg::DVec chi_inv(chi);
    dg::blas1::transform( chi, chi_inv, dg::INVERT<double>());
    dg::blas1::pointwiseDot( chi_inv, v2d, chi_inv);
    dg::DVec x_pcg = x;
    std::cout << "Create Polarisation object and set chi!\n";

    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> pol( grid, dg::not_normed, dg::centered, jfactor);
    pol.set_chi( chi);
    dg::Invert<dg::DVec > invert( x_pcg, p*p*Nx*Ny, eps);
    std::cout << "invert in #iterations "<< invert( pol, x_pcg, b, w2d, chi_inv, v2d) <<std::endl;
    //get all EV by decomposition
    dg::EVarbitraryMatrix pol_decomp(p*p*Nx*Ny);
    dg::DVec ev_top(2, 0.0), ev_bot(2, 0.0);
    pol_decomp( pol, ev_top, ev_bot);
    printvector<dg::DVec> (ev_top);
    printvector<dg::DVec> (ev_bot);
    return 0;
