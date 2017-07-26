#include <iostream>
#include <iomanip>

#include "blas.h"
#include "manualmatrix.h"

/* test manualmatrix.h
   where Ax = b is used as convention */

template< class Vector>
void printvector( const std::string& message, Vector& v)
{   std::cout<< message << " is of typeid: " << typeid(v).name() <<std::endl;
    for( uint i=0; i<v.size(); ++i)
    {   std::cout<<v[i]<<std::endl;
    }
    std::cout<<"- - - - - - - -"<<std::endl;
}

int main()
{   int n;
    double seed;
    std::cout << "Type n & seed! \n";
    std::cin >> n >> seed;

    dg::DVec x(n), b(n);
    for( int i = 0; i<n; ++i)
    {   x[i] = 0.0;
        b[i] = 1.0;
    }
    dg::RandPSDmatrix<dg::DVec> spd(n, seed);
    // does the Eigen based implementation really solve Ax = b?
    spd.lsolve( b, x);
    spd.symv(x, b);
    printvector<dg::DVec>("b needs to be all 1.0", b);
    return 0;
}
