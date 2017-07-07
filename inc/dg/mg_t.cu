#include <iostream>
#include <typeinfo>
#include "blas.h"
#include "manualmatrix.h"
#include "dg/backend/timer.cuh"
#include "eve.h"
#include "chebyshev.h"
#include <cmath>

template< class Vector>
void printvector( const std::string& message, Vector& v)
{   std::cout<< message << " is of typeid: " << typeid(v).name() <<std::endl;
    for( uint i=0; i<v.size(); ++i)
    {   std::cout<<v[i]<<std::endl;
    }
    std::cout<<"- - - - - - - -"<<std::endl;
}


int main()
{   int n, max_iter;/* n=#elements -> n-vector & nxn-matrix */
    double eps;
    std::cout << "Type n, eps, max_iter\n";
    std::cin >> n >> eps >> max_iter;
    const double h = 2.*M_PI/(n-1);
    //    dg::RandPSDmatrix<dg::HVec> A(n, 1.0);
    dg::Laplace1D<dg::HVec> A(n, 1./(h*h));
    dg::HVec b(n);
    for (int i=0; i<n; ++i)
    {   b[i] = std::sin(i*h);
    }

//    std::srand(time(NULL));
//    for (int i=0; i<n; ++i)
//    {   b[i] = std::rand()%100;
//    }
//    printvector( "b", b);
    dg::HVec ev = A.get_eigenvalues();
//    printvector( "ev", ev);
    double ev_min = *std::min_element(ev.begin(),ev.end());
    double ev_max = *std::max_element(ev.begin(),ev.end());
    std::cout << "Eigen max(EV) " << ev_max << std::endl;
//    dg::HVec x_pcg(n);
//    std::fill(x_pcg.begin(), x_pcg.end(), 0.0);
//    dg::eCG<dg::HVec> cg( x_pcg, max_iter);
//    double ev_ecg;
//    std::cout<< "Number of cg iterations " << cg( L, x_pcg, b, ev_ecg, eps) <<std::endl;
//    std::cout << ev_ecg << std::endl;
//    printvector( "x by CG", x_pcg);
    dg::HVec x_cheb(n);
    std::fill(x_cheb.begin(), x_cheb.end(), 0.0);
    dg::Chebyshev<dg::HVec> cheb( x_cheb, max_iter);
    std::cout << "#iterations " << cheb( A, x_cheb, b, ev_max, ev_min, eps) << std::endl;;
    printvector( "x by Chebyshev", x_cheb);
    dg::HVec x_eigen(n);
    A.lsolve(b, x_eigen);
    printvector( "x by Eigen", x_eigen);
    return 0;
}
/*
dg::Timer t;
t.tic();
t.toc();
std::cout << t.diff() << std::endl;
*/
