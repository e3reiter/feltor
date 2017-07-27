#include <iostream>

#include "blas.h"
#include "functors.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <spectra/SymEigsSolver.h>
#include <spectra/GenEigsSolver.h>
#include <spectra/MatOp/SparseGenMatProd.h>
#include "backend/timer.cuh"
using namespace Spectra;
/* Analyse any Matrix used in FELTOR
   ... not elegant, very slow but useful! */

namespace dg
{

template< class Vector>
class AnalysisMatrix
{
public:
    typedef typename VectorTraits<Vector>::value_type value_type;
    AnalysisMatrix() {}
    AnalysisMatrix( long int Mx, long int My): Mx_(Mx), My_(My) {}
    template< class Matrix>
    unsigned operator()( Matrix &A, long int div, value_type& ev_max);
    long int Mx_, My_;
};
template< class Vector>
template< class Matrix>
unsigned AnalysisMatrix<Vector>::operator()( Matrix& A, long int div, value_type& ev_max)
{   dg::Timer t;
    long int i_old = 0, ncv = Mx_/div;
    value_type zero = 0.0, one = 1.0;
    Vector vec(Mx_, zero), res(Mx_);
    std::cout<< "construct actual matrix..." <<std::endl;
    //Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> actualmatrix(Mx_, My_);
    Eigen::SparseMatrix<value_type> actualmatrix_sparse;//(Mx_, My_);
    Eigen::VectorXi vallo(My_);
    double value;
    std::vector<double> tvec;
    std::vector<long int> iidx, yidx;
    t.tic();
    for( long int i=0; i<My_; ++i)
    {   vec[i_old] = zero;
        vec[i] = one;
        dg::blas2::symv(A, vec, res);
        long int nnzero = 0;
        for( long int j=0; j<Mx_; ++j)
        {   value = res[j];
            if( value != 0.0)
            {   tvec.push_back(value);
                iidx.push_back(i);
                yidx.push_back(j);
                nnzero += 1;
            }
        }
        vallo(i) = Mx_ -nnzero;
        i_old=i;
    }
    t.toc();
    std::cout<< "first run" <<std::endl;
    std::cout<< "... took: "<< t.diff() <<std::endl;    t.tic();
    std::cout<< "#double "<< tvec.size() <<std::endl;
    actualmatrix_sparse.reserve(vallo);
    for( long int i=0; i<tvec.size(); ++i)
    {    actualmatrix_sparse.insert(iidx[i],yidx[i]) = tvec[i];
    }
    actualmatrix_sparse.makeCompressed();
    t.toc();
    std::cout<< "... took: "<< t.diff() <<std::endl;



//    std::cout<< "- - - - - - - - - - - - - -" <<std::endl;
//    std::cout<< "Eigen: all Eigenvalues ... " <<std::endl;
//    t.tic();
//    Eigen::SelfAdjolong intEigenSolver< Eigen::Matrix<value_type, Eigen::Dynamic, Eigen::Dynamic> > es_eigen(actualmatrix);
    Eigen::VectorXd ev ;//= es_eigen.eigenvalues().real();
//    t.toc();
//    ev_max = ev.maxCoeff();
//    std::cout<< "... EV: "<< ev_max <<std::endl;
//    std::cout<< "... took: "<< t.diff() <<std::endl;
//    std::cout<< "- - - - - - - - - - - - - - - - - -" <<std::endl;
//    std::cout<< "Spectra: max Eigenvalue, dense ..." <<std::endl;
//    DenseSymMatProd<double> op_dense(actualmatrix);
//    SymEigsSolver< double, LARGEST_ALGE, DenseSymMatProd<double> > es_densespectra(&op_dense, 1, ncv);
//    es_densespectra.init();
//    es_densespectra.compute();
//    t.toc();
//    if(es_densespectra.info() == SUCCESSFUL)
//      ev = es_densespectra.eigenvalues();
//    ev_max = ev[0];
//    std::cout<< "... EV: "<< ev_max <<std::endl;
//    std::cout<< "... took: "<< t.diff() <<std::endl;
//    std::cout<< "- - - - - - - - - - - - - - - - - -" <<std::endl;
//    std::cout<< "Sparsify ..." <<std::endl;
//    Eigen::SparseMatrix<double> actualmatrix_sparse = actualmatrix.sparseView();
    std::cout<<actualmatrix_sparse.nonZeros()<<" "<<actualmatrix_sparse.isCompressed()<<std::endl;
    std::cout<< "- - - - - - - - - - - - - - - - - -" <<std::endl;
    std::cout<< "Spectra: max Eigenvalue, sparse ..." <<std::endl;
    t.tic();
    SparseGenMatProd<double> op_sparse(actualmatrix_sparse);
    SymEigsSolver< double, LARGEST_ALGE, SparseGenMatProd<double> > es_sparsespectra(&op_sparse, 1, ncv);
    es_sparsespectra.init();
    es_sparsespectra.compute();
    t.toc();
    if(es_sparsespectra.info() == SUCCESSFUL)
        ev = es_sparsespectra.eigenvalues();
    ev_max = ev[0];
    std::cout<< "... EV: "<< ev_max <<std::endl;
    std::cout<< "...took: "<< t.diff() <<std::endl;
    return 0;
}

}
