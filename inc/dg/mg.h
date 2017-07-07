#ifndef _DG_MG_
#define _DG_MG_

#include <cmath>

#include "blas.h"
#include "functors.h"

#ifdef DG_BENCHMARK
#include "backend/timer.cuh"
#endif //DG_BENCHMARK

/*!@file
 * Conjugate gradient class and functions
 */

namespace dg
{

//// TO DO: check for better stopping criteria using condition number estimates

/**
* @brief Functor class for the preconditioned conjugate gradient method to solve
* \f[ Ax=b\f]
*
 @ingroup invert
 @tparam Vector The Vector class: needs to model Assignable

 The following 3 pseudo - BLAS routines need to be callable
 \li value_type dot = dg::blas1::dot( const Vector&, const Vector&);
 \li dg::blas1::axpby();  with the Vector type
 \li dg::blas2::symv(Matrix& m, Vector1& x, Vector2& y ); with the Matrix type
 \li value_type dot = dg::blas2::dot( );  with the Preconditioner type
 \li dg::blas2::symv( ); with the Preconditioner type

 @note Conjugate gradients might become unstable for positive semidefinite
 matrices arising e.g. in the discretization of the periodic laplacian
*/
template< class Vector>
class pCG
{
public:
    typedef typename VectorTraits<Vector>::value_type value_type;//!< value type of the Vector class
    /**
     * @brief Allocate nothing,
     */
    pCG() {}
    /**
     * @brief Reserve memory for the pcg method
     *
     * @param copyable A Vector must be copy-constructible from this
     * @param max_iter Maximum number of iterations to be used
     */
    pCG( const Vector& copyable, unsigned max_iter):r(copyable), p(r), ap(r), max_iter(max_iter) {}
    /**
     * @brief Set the maximum number of iterations
     *
     * @param new_max New maximum number
     */
    void set_max( unsigned new_max)
    {   max_iter = new_max;
    }
    /**
     * @brief Get the current maximum number of iterations
     *
     * @return the current maximum
     */
    unsigned get_max() const
    {   return max_iter;
    }

    /**
     * @brief Set internal storage and maximum number of iterations
     *
     * @param copyable
     * @param max_iterations
     */
    void construct( const Vector& copyable, unsigned max_iterations)
    {   ap = p = r = copyable;
        max_iter = max_iterations;
    }
    /**
     * @brief Solve the system A*x = b using a preconditioned conjugate gradient method
     *
     * The iteration stops if \f$ ||Ax|| < \epsilon( ||b|| + C) \f$ where \f$C\f$ is
     * a correction factor to the absolute error
     @tparam Matrix The matrix class: no requirements except for the
            BLAS routines
     @tparam Preconditioner no requirements except for the blas routines. Thus far the dg library
        provides only diagonal preconditioners, which should be enough if the result is extrapolated from
        previous timesteps.
     * In every iteration the following BLAS functions are called: \n
       symv 1x, dot 1x, axpby 2x, Prec. dot 1x, Prec. symv 1x
     * @param A A symmetric positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param eps The relative error to be respected
     * @param nrmb_correction Correction factor C for norm of b
     * @attention This versions uses the Preconditioner to compute the norm for the error condition (this safes one scalar product)
     *
     * @return Number of iterations used to achieve desired precision
     */
    template< class Matrix>
    unsigned operator()( Matrix& A, Vector& x, const Vector& b, value_type eps = 1e-12, value_type nrmb_correction = 1);
    //version of pCG where Preconditioner is not trivial
    /**
     * @brief Solve the system A*x = b using a preconditioned conjugate gradient method
     *
     * The iteration stops if \f$ ||Ax||_S < \epsilon( ||b||_S + C) \f$ where \f$C\f$ is
     * a correction factor to the absolute error and \f$ S \f$ defines a square norm
     @tparam Matrix The matrix class: no requirements except for the
            BLAS routines
     @tparam Preconditioner no requirements except for the blas routines. Thus far the dg library
        provides only diagonal preconditioners, which should be enough if the result is extrapolated from
        previous timesteps.
     @tparam SquareNorm  (usually is the same as the container class)

     * In every iteration the following BLAS functions are called: \n
       symv 1x, dot 1x, axpby 2x, Prec. dot 1x, Prec. symv 1x
     * @param A A symmetric positive definit matrix
     * @param x Contains an initial value on input and the solution on output.
     * @param b The right hand side vector. x and b may be the same vector.
     * @param P The preconditioner to be used
     * @param S Weights used to compute the norm for the error condition
     * @param eps The relative error to be respected
     * @param nrmb_correction Correction factor C for norm of b
     *
     * @return Number of iterations used to achieve desired precision
     */
private:
    Vector r, p, ap;
    unsigned max_iter;
};

/*
    compared to unpreconditioned compare
    dot(r,r), axpby()
    to
    dot( r,P,r), symv(P)
    i.e. it will be slower, if P needs to be stored
    (but in our case P_{ii} can be computed directly
    compared to normal preconditioned compare
    ddot(r,P,r), dsymv(P)
    to
    ddot(r,z), dsymv(P), axpby(), (storage for z)
    i.e. it's surely faster if P contains no more elements than z
    (which is the case for diagonal scaling)
    NOTE: the same comparison hold for A with the result that A contains
    significantly more elements than z whence ddot(r,A,r) is far slower than ddot(r,z)
*/
template< class Vector>
template< class Matrix>
unsigned pCG< Vector>::operator()( Matrix& A, Vector& x, const Vector& b, value_type eps, value_type nrmb_correction)
{   value_type nrmb = sqrt( blas1::dot( b, b));
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if(rank==0)
#endif //MPI
    {   std::cout << "Norm of b "<<nrmb <<"\n";
        std::cout << "Residual errors: \n";
    }
#endif //DG_DEBUG
    if( nrmb == 0)
    {   blas1::axpby( 1., b, 0., x);
        return 0;
    }
    blas2::symv( A, x, r);
    blas1::axpby( 1., b, -1., r);
    blas1::axpby(1., r, 0., p);
    //note that dot does automatically synchronize
    value_type nrm2r_old = blas1::dot( r,r);
    if( sqrt( nrm2r_old ) < eps*(nrmb + nrmb_correction)) //if x happens to be the solution
        return 0;
    value_type alpha, nrm2r_new;
    for( unsigned i=1; i<max_iter; i++)
    {   blas2::symv( A, p, ap);
        alpha = nrm2r_old /blas1::dot( p, ap);
        blas1::axpby( alpha, p, 1.,x);
        blas1::axpby( -alpha, ap, 1., r);
        nrm2r_new = blas1::dot( r, r);
#ifdef DG_DEBUG
#ifdef MPI_VERSION
        if(rank==0)
#endif //MPI
        {   std::cout << "Absolute "<<sqrt( nrm2r_new) <<"\t ";
            std::cout << " < Critical "<<eps*nrmb + eps <<"\t ";
            std::cout << "(Relative "<<sqrt( nrm2r_new)/nrmb << ")\n";
        }
#endif //DG_DEBUG
        if( sqrt( nrm2r_new) < eps*(nrmb + nrmb_correction))
            return i;
        blas1::axpby(1., r, nrm2r_new/nrm2r_old, p);//blas2::symv(1.,P, r, nrm2r_new/nrm2r_old, p );
        nrm2r_old=nrm2r_new;
    }
    return max_iter;
}
} //namespace dg
#endif //_DG_MG_
