/* test eigenmatrix.h for manualmatrix/FELTOR matrices */

#include <iostream>
#include "manualmatrix.h"
#include "eigenmatrix.h"
#include "elliptic.h"

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

template< class Vector>
void printvector( Vector& v)
{ for( uint i=0; i<v.size(); ++i)
  {   std::cout<<v[i]<<std::endl;
  }
  std::cout<<"- - - - - - - -"<<std::endl;
}

int main()
{   // for some random symmetric positive matrix
    int n_rand, nev_top, nev_bot;
    std::cout<< "n for random psd and number of EVtop/EVbot, please:" <<std::endl;
    std::cin>> n_rand >> nev_top >> nev_bot;
    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
    std::cout<< "constructing random psd ..." <<std::endl;
    dg::RandPSDmatrix<dg::DVec> spd(n_rand, 1.1);
    dg::EVarbitraryMatrix am(n_rand);
    dg::DVec ev_top(nev_top, 0.0), ev_bot(nev_bot, 0.0);
    am( spd, ev_top, ev_bot);
    printvector<dg::DVec> (ev_top);
    printvector<dg::DVec> (ev_bot);
    // now for the elliptic object
    unsigned p, Nx, Ny;
    double eps, jfactor;
    std::cout<< "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -" <<std::endl;
    std::cout << "Type p, Nx and Ny and epsilon and jfactor (1), #EVtop/EVbot too! \n";
    std::cin >> p >> Nx >> Ny;
    std::cin >> eps >> jfactor >> nev_top >> nev_bot;
    dg::Grid2d grid( 0, lx, 0, ly, p, Nx, Ny, bcx, bcy);
    dg::DVec chi =  dg::evaluate( pol, grid);
    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
    std::cout << "Create Polarisation object and set chi!\n";
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> ell( grid, dg::not_normed, dg::centered, jfactor);
    ell.set_chi( chi);
    dg::EVarbitraryMatrix ae( p*p*Nx*Ny);
    dg::DVec evell_top(nev_top, 0.0), evell_bot(nev_bot, 0.0);
    ae( ell, evell_top, evell_bot);
    printvector<dg::DVec> (evell_top);
    printvector<dg::DVec> (evell_bot);
    return 0;
}
