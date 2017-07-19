#include "blas.h"
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
//#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace dg
{
/**
 *@brief Matrix class providing dense positive semi-definite (pseudo-)random test matrix.
 *
 * The dense random psd matrix is generated calculating A'*A for some random A with
 * positive entries only. It's (too) well conditioned at the moment.
 *
 * @tparam Vector The Vector class to use
 * This class has the SelfMadeMatrixTag so it can be used in blas2::symv functions
 * and thus in a conjugate gradient solver - WARNING: slow (for testing only).
 */
template<class Vector>
class RandPSDmatrix
{   int n_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> psd;
    Eigen::VectorXd ev_;
  double ev_max_;
    Vector eigenvalues_;
public:
    /**
     *@brief Construct for n rows & n columns and calculates eigenvalues
     *
     *@param n #rows & #columns
     */
    RandPSDmatrix(const int n, const double seed):n_(n), eigenvalues_(n)
    {   std::srand(seed);
        {   Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> randposmat(n_, n_);
            for( int i=0; i<n_; ++i)
            {   for( int j=0; j<n_; ++j)
                {   randposmat(i,j) = std::rand()%2;
                }
            }
            psd = randposmat*randposmat.transpose();
            //            std::cout << psd << std::endl;
        }
        ev_ = psd.eigenvalues().real();
        ev_max_ = ev_.maxCoeff();
        transfer(ev_, eigenvalues_);
    }
    const Vector& get_eigenvalues() const
    {   return eigenvalues_;
    }
    const double& get_maxev() const
    {   return ev_max_;
    }

    /**
     *@brief Solve linear System PSDx=b.
     *
     * Use "a robust Cholesky decomposition of a positive semidefinite [...] matrix" from the
     * Eigen library to solve the linear system PSDx=b.
     */
    void lsolve( const Vector& b, Vector& x)
    {   Eigen::VectorXd b_(n_);
        transfer( b, b_);
        Eigen::VectorXd x_ = psd.ldlt().solve(b_);
        transfer( x_, x);
    }
    template<class V1, class V2>
    void transfer(const V1& v1, V2& v2)
    {   for( uint i=0; i<v1.size(); ++i)
        {   v2[i] = v1[i];
        }
    }
    /**
     *@brief Computes Matrix-Vector Product PSDx=y
     *
     *@param x left hand side
     *@param y result
     */
   void symv(const Vector& x, Vector& y)
    {   std::fill(y.begin(), y.end(), 0.0);
        for( int i=0; i<n_; ++i)
        {   for( int j=0; j<n_; ++j)
            {   y[i] += psd(i,j)*x[j];
            }
        }
    }
};

template<class Vector>
class Laplace1D
{   int n_;
    double h_;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> l1d;
    Eigen::VectorXd ev_;
    Vector eigenvalues_;
public:
    /**
     *@brief Construct for n rows & n columns and calculates eigenvalues
     *
     *@param n #rows & #columns
     */
  Laplace1D(const int n, const double h):n_(n), h_(h), eigenvalues_(n)
  {   l1d.resize(n_, n_);
      l1d.setZero();
      l1d(0,0) = 2./h;
      l1d(0,1) = -1./h;
      for( int i = 1; i<n-1; ++i)
        { l1d(i,i-1) = -1./h;
          l1d(i,i)   =  2./h;
          l1d(i,i+1) = -1./h;
        }
      l1d(n-1,n-2) = -1./h;
      l1d(n-1,n-1) = 2./h;
      //      std::cout << l1d << std::endl;
      ev_ = l1d.eigenvalues().real();
      transfer(ev_, eigenvalues_);
    }
    const Vector& get_eigenvalues() const
    {   return eigenvalues_;
    }
    /**
     *@brief Solve linear System PSDx=b.
     *
     * Use "a robust Cholesky decomposition of a positive semidefinite [...] matrix" from the
     * Eigen library to solve the linear system PSDx=b.
     */
    void lsolve( const Vector& b, Vector& x)
    {   Eigen::VectorXd b_(n_);
        transfer( b, b_);
        Eigen::VectorXd x_ = l1d.colPivHouseholderQr().solve(b_);
        transfer( x_, x);
    }
    template<class V1, class V2>
    void transfer(const V1& v1, V2& v2)
    {   for( uint i=0; i<v1.size(); ++i)
        {   v2[i] = v1[i];
        }
    }
    /**
     *@brief Computes Matrix-Vector Product PSDx=y
     *
     *@param x left hand side
     *@param y result
     */
    void symv(const Vector& x, Vector& y)
    {   std::fill(y.begin(), y.end(), 0.0);
        for( int i=0; i<n_; ++i)
        {   for( int j=0; j<n_; ++j)
            {   y[i] += l1d(i,j)*x[j];
            }
        }
    }
};
///@cond
template< class V>
struct MatrixTraits< RandPSDmatrix<V> >
{   typedef typename VectorTraits<V>::value_type value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
  template< class V>
  struct MatrixTraits< Laplace1D<V> >
  {   typedef typename VectorTraits<V>::value_type value_type;
    typedef SelfMadeMatrixTag matrix_category;
  };
///@endcond
} //namespace test
