#include <iostream>
#include <typeinfo>
#include <cmath>
#include "blas.h"
#include "elliptic.h"
#include "eve.h"
#include "chebyshev.h"
#include "cg.h"
#include "backend/timer.cuh"


template< class Vector>
void printvector( const std::string& message, Vector& v)
{   std::cout<< message << " is of typeid: " << typeid(v).name() <<std::endl;
    for( uint i=0; i<v.size(); ++i)
    {   std::cout<<v[i]<<std::endl;
    }
    std::cout<<"- - - - - - - -"<<std::endl;
}

/* Test problem construction */
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
double sol(double x, double y)
{   return sin( x)*sin(y);
}
double der(double x, double y)
{   return cos( x)*sin(y);
}

/* create a number of grids, h, 2h, 4h, solve and compare solution & afford & maximum EV.

 */
int main()
{   return 0;
}
