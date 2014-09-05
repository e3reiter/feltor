#include <iostream>

#include <cusp/print.h>
#include <mpi.h>

#include "mpi_evaluation.h"
#include "mpi_precon.h"
#include "dz.h"
#include "functions.h"
#include "../blas2.h"
#include "../functors.h"
#include "interpolation.cuh"

#include "mpi_init.h"

struct Field
{
    Field( double R_0, double I_0):R_0(R_0), I_0(I_0){}
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp)
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {
            double gradpsi = ((y[0][i]-R_0)*(y[0][i]-R_0) + y[1][i]*y[1][i])/I_0/I_0;
            yp[2][i] = y[0][i]*sqrt(1 + gradpsi);
            yp[0][i] = y[0][i]*y[1][i]/I_0;
            yp[1][i] = -y[0][i]*y[0][i]/I_0 + R_0/I_0*y[0][i] ;
        }
    }
    private:
    double R_0, I_0;
};

double R_0 = 10;
double I_0 = 40;
double func(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    return r2*sin(phi);
}
double deri(double R, double Z, double phi)
{
    double r2 = (R-R_0)*(R-R_0)+Z*Z;
    return I_0/R/sqrt(I_0*I_0 + r2)* r2*cos(phi);
}


int main(int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm;
    mpi_init3d( dg::PER, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    Field field( R_0, I_0);
    dg::MPI_Grid3d g3d( R_0 - 1, R_0+1, -1, 1, 0, 2.*M_PI, n, Nx, Ny, Nz, comm);
    const dg::MPI_Precon w3d = dg::create::weights( g3d);
    dg::DZ<dg::MMatrix, dg::MVec> dz( field, g3d);

    dg::MVec function = dg::evaluate( func, g3d), derivative(function);
    const dg::MVec solution = dg::evaluate( deri, g3d);
    dz( function, derivative);
    dg::blas1::axpby( 1., solution, -1., derivative);
    double norm = dg::blas2::dot( w3d, solution);
    if(rank==0)std::cout << "Norm Solution "<<sqrt( norm)<<"\n";
    double norm2 = sqrt( dg::blas2::dot( derivative, w3d, derivative)/norm);
    if(rank==0)std::cout << "Relative Difference Is "<< norm2<<"\n";    
    MPI_Finalize();
    return 0;
}