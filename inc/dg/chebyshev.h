#ifndef _DG_CHEB_
#define _DG_CHEB_

#include <cmath>

#include "blas.h"
#include "functors.h"

#ifdef DG_BENCHMARK
#include "backend/timer.cuh"
#endif //DG_BENCHMARK


namespace dg
{
template< class Vector>
class Chebyshev
{
public:
    typedef typename VectorTraits<Vector>::value_type value_type;//!< value type of the Vector class
    /**
     * @brief Allocate nothing,
     */
    Chebyshev() {}
    Chebyshev( const Vector& copyable, unsigned grade): r(copyable), p(r), ap(r), grade(grade) {}
    void set_grade( unsigned new_grade)
    {   grade = new_grade;
    }
    unsigned get_grade() const
    {   return grade;
    }
    void construct( const Vector& copyable, unsigned grade_chebychev)
    {   ap = p = r = copyable;
        grade = grade_chebychev;
    }
    template< class Matrix>
    int operator()( Matrix& A, Vector& x, const Vector& b, value_type max_ev, value_type min_ev, value_type eps = 1e-6, value_type nrmb_correction = 1.);
private:
    Vector r, p, ap;
    unsigned grade;
};
template< class Vector>
template< class Matrix>
int Chebyshev< Vector>::operator()( Matrix& A, Vector& x, const Vector& b, value_type max_ev, value_type min_ev, value_type eps, value_type nrmb_correction)
{   value_type theta = ( max_ev+min_ev) /2.;
    value_type delta = ( max_ev-min_ev) /2.;
    // initialise
    value_type nrmb = sqrt( blas1::dot( b, b));
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
    blas2::symv( A, x, r);
    blas1::axpby( 1., b, -1., r);
    value_type nrm2r = blas1::dot( r,r);               // needed just to compare with CG
    if( sqrt( nrm2r) < eps*( nrmb + nrmb_correction))   // ~
      return 0;                                        // ~
    blas1::axpby( 1./theta, r, 0., p);
    value_type sigmadash = 2. *theta/delta;
    value_type deltadash = 2. /delta;
    value_type rhold = delta/theta;
    value_type rho;
    // apply chebyshev
    for( unsigned i=0; i<grade; ++i)
    {   blas1::axpby( 1., p, 1., x);
        blas2::symv( A, p, ap);
        blas1::axpby( -1., ap, 1., r);
        nrm2r = blas1::dot( r,r);                          // needed just to compare with CG
#ifdef DG_DEBUG
#ifdef MPI_VERSION
        if(rank==0)
#endif //MPI
          {
            std::cout << "Absolute "<<sqrt( nrm2r) <<"\t ";
            std::cout << " < Critical "<<eps*nrmb + eps <<"\t ";
            std::cout << "(Relative "<<sqrt( nrm2r)/nrmb << ")\n";
          }
#endif //DG_DEBUG

        if( sqrt( nrm2r) < eps*( nrmb + nrmb_correction))   // ~
          return i;                                        // ~
        rho = 1. /( sigmadash -rhold);
        blas1::axpby( deltadash*rho, r, rhold*rho, p);
        rhold = rho;
    }
    return grade;
};
template<class container>
struct cInvert
{   typedef typename VectorTraits<container>::value_type value_type;

    /**
     * @brief Allocate nothing
     *
     */
    cInvert()
    {   multiplyWeights_ = true;
        set_extrapolationType(2);
        nrmb_correction_ = 1.;
    }

    /**
     * @brief Constructor
     *
     * @param copyable Needed to construct the two previous solutions
     * @param max_iter maximum iteration in conjugate gradient
     * @param eps relative error in conjugate gradient
     * @param extrapolationType number of last values to use for extrapolation of the current guess
     * @param multiplyWeights if true the rhs shall be multiplied by the weights before cg is applied
     * @param nrmb_correction Correction factor for norm of b (cf. CG)
     */
    cInvert(const container& copyable, unsigned max_iter, value_type eps, int extrapolationType = 2, bool multiplyWeights = true, value_type nrmb_correction = 1)
    {   construct( copyable, max_iter, eps, extrapolationType, multiplyWeights, nrmb_correction);
    }

    /**
     * @brief to be called after default constructor
     *
     * @param copyable Needed to construct the two previous solutions
     * @param max_iter maximum iteration in conjugate gradient
     * @param eps relative error in conjugate gradient
     * @param extrapolationType number of last values to use for extrapolation of the current guess
     * @param multiplyWeights if true the rhs shall be multiplied by the weights before cg is applied
     * @param nrmb_correction Correction factor for norm of b (cf. CG)
     */
    void construct( const container& copyable, unsigned max_iter, value_type eps, int extrapolationType = 2, bool multiplyWeights = true, value_type nrmb_correction = 1.)
    {   set_size( copyable, max_iter);
        set_accuracy( eps, nrmb_correction);
        multiplyWeights_=multiplyWeights;
        set_extrapolationType( extrapolationType);
    }


    /**
     * @brief Set vector storage and maximum number of iterations
     *
     * @param assignable
     * @param max_iterations
     */
    void set_size( const container& assignable, unsigned max_iterations)
    {   cheb.construct(assignable, max_iterations);
        phi0 = phi1 = phi2 = assignable;
    }
    /**
     * @brief Set accuracy parameters for following inversions
     *
     * @param eps
     * @param nrmb_correction
     */
    void set_accuracy( value_type eps, value_type nrmb_correction = 1.)
    {   eps_ = eps;
        nrmb_correction_ = nrmb_correction;
    }
    /**
     * @brief Set the extrapolation Type for following inversions
     *
     * @param extrapolationType number of last values to use for next extrapolation of initial guess
     */
    void set_extrapolationType( int extrapolationType)
    {   assert( extrapolationType <= 3 && extrapolationType >= 0);
        switch(extrapolationType)
        {   case(0):
                alpha[0] = 0, alpha[1] = 0, alpha[2] = 0;
                break;
            case(1):
                alpha[0] = 1, alpha[1] = 0, alpha[2] = 0;
                break;
            case(2):
                alpha[0] = 2, alpha[1] = -1, alpha[2] = 0;
                break;
            case(3):
                alpha[0] = 3, alpha[1] = -3, alpha[2] = 1;
                break;
            default:
                alpha[0] = 2, alpha[1] = -1, alpha[2] = 0;
        }
    }
    /**
     * @brief Set the maximum number of iterations
     *
     * @param new_max New maximum number
     */
    void set_max( unsigned new_max)
    {   cheb.set_grade( new_max);
    }
    /**
     * @brief Get the current maximum number of iterations
     *
     * @return the current maximum
     */
    unsigned get_max() const
    {   return cheb.get_grade();
    }
    /**
    * @brief Return last solution
    */
    const container& get_last() const
    {   return phi0;
    }
    /**
     * @brief Solve linear problem
     *
     * Solves the Equation \f[ \hat O \phi = W\rho \f] using a preconditioned
     * conjugate gradient method. The initial guess comes from an extrapolation
     * of the last solutions
     * @tparam SymmetricOp Symmetric operator with the SelfMadeMatrixTag
        The functions weights() and precond() need to be callable and return
        weights and the preconditioner for the conjugate gradient method.
        The Operator is assumed to be symmetric!
     * @param op selfmade symmetric Matrix operator class
     * @param phi solution (write only)
     * @param rho right-hand-side
     * @note computes inverse weights from the weights
     *
     * @return number of iterations used
     */
//    template< class SymmetricOp >
//    unsigned operator()( SymmetricOp& op, container& phi, const container& rho)
//    {   container inv_weights( op.weights());
//        dg::blas1::transform( inv_weights, inv_weights, dg::INVERT<double>());
//       return this->operator()(op, phi, rho, op.weights(), op.precond(), inv_weights);
//    }

    /**
     * @brief Solve linear problem
     *
     * Solves the Equation \f[ \hat O \phi = W\rho \f] using a preconditioned
     * conjugate gradient method. The initial guess comes from an extrapolation
     * of the last solutions.
     * @tparam SymmetricOp Symmetric matrix or operator (with the selfmade tag)
     * @tparam Preconditioner class of the Preconditioner
     * @param op selfmade symmetric Matrix operator class
     * @param phi solution (write only)
     * @param rho right-hand-side
     * @param w The weights that made the operator symmetric
     * @param p The preconditioner
     * @note computes inverse weights from the weights
     *
     * @return number of iterations used
     */
//    template< class SymmetricOp, class Preconditioner >
//    unsigned operator()( SymmetricOp& op, container& phi, const container& rho, const container& w, Preconditioner& p)
//    {   container inv_weights( w);
//        dg::blas1::transform( inv_weights, inv_weights, dg::INVERT<double>());
//        return this->operator()(op, phi, rho, w, p, inv_weights);
//    }
//
//    /**
//     * @brief Solve linear problem
//     *
//     * Solves the Equation \f[ \hat O \phi = W\rho \f] using a preconditioned
//     * conjugate gradient method. The initial guess comes from an extrapolation
//     * of the last solutions.
//     * @tparam SymmetricOp Symmetric matrix or operator (with the selfmade tag)
//     * @tparam Preconditioner class of the Preconditioner
//     * @param op selfmade symmetric Matrix operator class
//     * @param phi solution (write only)
//     * @param rho right-hand-side
//     * @param w The weights that made the operator symmetric
//     * @param p The preconditioner
//     * @param inv_weights The inverse weights used to compute the scalar product in the CG solver
//     * @note Calls the most general form of the CG solver with SquareNorm being the container class
//     *
//     * @return number of iterations used
//     */
    template< class SymmetricOp, class Preconditioner >
    unsigned operator()( SymmetricOp& op, container& phi, const container& rho,  value_type ev_max, value_type ev_min, const container& w, Preconditioner& p, const container& inv_weights)
    {   assert( phi0.size() != 0);
        assert( &rho != &phi);
        blas1::axpby( alpha[0], phi0, alpha[1], phi1, phi); // 1. phi0 + 0.*phi1 = phi
        blas1::axpby( alpha[2], phi2, 1., phi); // 0. phi2 + 1. phi0 + 0.*phi1 = phi
        unsigned number;
#ifdef DG_BENCHMARK
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        Timer t;
        t.tic();
#endif //DG_BENCHMARK
        if( multiplyWeights_ )
        {   dg::blas2::symv( w, rho, phi2);
          number = cheb( op, phi, phi2, ev_max, ev_min, eps_, nrmb_correction_);
        }
        else
          number = cheb( op, phi, rho, ev_max, ev_min, eps_, nrmb_correction_);
#ifdef DG_BENCHMARK
#ifdef MPI_VERSION
        if(rank==0)
#endif //MPI
            std::cout << "# of cheb iterations \t"<< number << "\t";
        t.toc();
#ifdef MPI_VERSION
        if(rank==0)
#endif //MPI
            std::cout<< "took \t"<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
        phi1.swap( phi2);
        phi0.swap( phi1);
        blas1::axpby( 1., phi, 0, phi0);
        return number;
    }

private:
    value_type eps_, nrmb_correction_;
    container phi0, phi1, phi2;
    dg::Chebyshev< container > cheb;
    value_type alpha[3];
    bool multiplyWeights_;
};
} //namespace dg
#endif // _DG_CHEB_

//  value_type alpha = (max_ev+min_ev)/2.;
//  value_type c =  (max_ev-min_ev)/2.;
//  blas2::symv( A, x, r);
//  blas1::axpby( 1., b, -1., r);
//  blas1::axpby( 1., r, 0., p);
//  blas1::axpby( 1./alpha, p, 1., x);
//  blas2::symv( A, x, r);
//  blas1::axpby( 1., b, -1., r);
//
//  value_type psi = -0.5*c*c/(alpha*alpha);
//  value_type omega = 1./(alpha - c*c/(2*alpha));
//  blas1::axpby( 1., r, -psi, p);
//  blas1::axpby( omega, p, 1., x);
//  blas2::symv( A, x, r);
//  blas1::axpby( 1., b, -1., r);
//
//  for( int i=0; i<grade; ++i)
//  {   psi = -c*c/4*omega*omega;
//      omega = 1./(alpha - c*c/4*omega);
//      blas1::axpby( 1., r, -psi, p);
//      blas1::axpby( omega, p, 1., x);
//      blas2::symv( A, x, r);
//      blas1::axpby( 1., b, -1., r);
//      if(blas1::dot( r, r) <1e-8)
//        return i;
//  }
//  return -1;
