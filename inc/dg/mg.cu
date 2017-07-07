#include <iostream>
#include "chebyshev.h"
#include "elliptic.h"
#include "backend/timer.cuh"

const double lx = M_PI;
const double ly = 2.*M_PI;

double fct(double x, double y)
{   return sin(y)*sin(x+M_PI/2.);
}
double derivative( double x, double y)
{   return cos(x+M_PI/2.)*sin(y);
}
double laplace_fct( double x, double y)
{   return 2*sin(y)*sin(x+M_PI/2.);
}
dg::bc bcx = dg::NEU;
double initial( double x, double y)
{   return sin(0);
}

int main()
{   dg::Timer t;
    unsigned n, Nx, Ny;
    std::cout << "Type n, Nx, Ny";
    std::cin >> n >> Nx >> Ny;
    dg::Grid2d grid( 0., lx, 0, ly, n, Nx, Ny, bcx, dg::PER);
    std::cout<<"Evaluate initial condition...\n";
    dg::DVec x = dg::evaluate( initial, grid);

    std::cout << "Create Laplacian...\n";
    t.tic();
    dg::DMatrix DX = dg::create::dx( grid);
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> lap( grid, dg::not_normed, dg::forward);
    t.toc();
    std::cout<< "Creation took "<<t.diff()<<"s\n";

    return 0;
}
