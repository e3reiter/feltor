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
    wrapper( int m, Matrix& A): m_(m), A_(A) {}
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
            x_[i] = x_in[i];
        A_.symv(x_, y_);
        for( int i=0; i<m_; ++i)
            y_out[i] = y_[i];
    }
private:
    int m_;
    Matrix &A_;
    Vector x_, y_;
};

// Calculates maximum Eigenvalue of arbitrary FELTOR Matrix
template< class Vector>
class EVarbitraryMatrix
{
public:
  typedef typename VectorTraits<Vector>::value_type value_type;
  EVarbitraryMatrix() {}
  EVarbitraryMatrix( int m, int p):m_(m), p_(p) {}
  template< class Matrix>
  void operator()( Matrix &A, value_type &ev_max);
private:
  int m_, p_;
};

template< class Vector>
template< class Matrix>
void EVarbitraryMatrix<Vector>::operator()( Matrix& A, value_type &ev_max)
{   dg::wrapper<Vector, Matrix> ae(m_, A);
    Spectra::SymEigsSolver<double, Spectra::LARGEST_ALGE, dg::wrapper<Vector, Matrix> > eigs(&ae, 1, p_);
    eigs.init();
    eigs.compute();
    Eigen::VectorXd evalues = eigs.eigenvalues();
    ev_max = evalues[0];
}

} //namespace dg
