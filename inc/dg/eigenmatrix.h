/* Calculate maximum Eigenvalue of arbitrary FELTOR Matrix
   -------------------------------------------------------

   Matrix object must provide symv(), than wrapper is used
   as interface to SpectrA functions (based on Eigen) in
   EVarbitraryMatrix to non-iteratively calculate the
   maximum Eigenvalue.

   Usecase: device=omp as its for testing multigrid only   */


#include <Eigen/Core>              //http://bitbucket.org/eigen/eigen/get/3.3.4.tar.bz2
#include <spectra/SymEigsSolver.h> //https://github.com/yixuan/spectra/archive/v0.5.0.tar.gz
namespace dg
{


// User-supplied matrix operation class for SpectrA matrix-
// free usage as in https://spectralib.org/quick-start.html
template<class Vector, class Matrix>
class wrapper
{
public:
    wrapper() {}
    wrapper(const int m, Matrix &A): m_(m), A_(A) {}
    int rows()
    {   return m_;
    }
    int cols()
    {   return m_;
    }
    void perform_op(double *x_in, double *y_out)
    {   x_.resize(m_);
        y_.resize(m_);
        for( int i=0; i<m_; ++i)
        {   x_[i] = x_in[i];
        }
        A_.symv(x_, y_);
        for( int i=0; i<m_; ++i)
        {   y_out[i] = y_[i];
        }
    }
private:
    int m_;
    Matrix &A_;
    Vector x_, y_;
};


// Calculates maximum Eigenvalue of arbitrary FELTOR Matrix
class EVarbitraryMatrix
{
private:
    int n_;
    template< class Vector, class Matrix>
    void ev_topdown( Matrix &A, Vector &ev, int nev, int ncv);
    template< class Vector, class Matrix>
    void ev_downtop( Matrix &A, Vector &ev, int nev, int ncv);
public:
    EVarbitraryMatrix() {}
    EVarbitraryMatrix(const int n):n_(n) {}
    template< class Vector, class Matrix>
    void operator()( Matrix &A, Vector &ev_top, Vector &ev_bot);
};

template< class Vector, class Matrix>
void EVarbitraryMatrix::ev_topdown( Matrix &A, Vector &ev, int nev, int ncv)
{   dg::wrapper<Vector, Matrix> ae(n_, A);
    Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, dg::wrapper<Vector, Matrix> > eigs(&ae, nev, ncv);
    eigs.init();
    eigs.compute();
    Eigen::VectorXd evalues = eigs.eigenvalues();
    for( int i=0; i<nev; ++i)
    {   ev[i] = evalues[i];
    }
}
template< class Vector, class Matrix>
void EVarbitraryMatrix::ev_downtop( Matrix &A, Vector &ev, int nev, int ncv)
{   dg::wrapper<Vector, Matrix> ae(n_, A);
    Spectra::SymEigsSolver<double, Spectra::SMALLEST_ALGE, dg::wrapper<Vector, Matrix> > eigs(&ae, nev, ncv);
    eigs.init();
    eigs.compute();
    Eigen::VectorXd evalues = eigs.eigenvalues();
    for( int i=0; i<nev; ++i)
    {   ev[i] = evalues[i];
    }
}

template< class Vector, class Matrix>
void EVarbitraryMatrix::operator()( Matrix& A, Vector &ev_top, Vector &ev_bot)
{   int nev_top = ev_top.size(), nev_bot = ev_bot.size();
    int ncv_top = nev_top+10, ncv_bot = nev_bot+100;
    if( ncv_top > n_)
    {   ncv_top = n_;
    }
    if( ncv_bot > n_)
    {   ncv_bot = n_;
    }
    if( nev_top > 0)
    {   EVarbitraryMatrix::ev_topdown( A, ev_top, nev_top, ncv_top);
    }
    if( nev_bot > 0)
    {   EVarbitraryMatrix::ev_downtop( A, ev_bot, nev_bot, ncv_bot);
    }
}


} //namespace dg
