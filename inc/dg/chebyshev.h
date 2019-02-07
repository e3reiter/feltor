#ifndef _DG_CHEB_
#define _DG_CHEB_

#include <cmath>

#include "blas.h"
#include "functors.h"


namespace dg
{
template< class Vector>
class Chebyshev
{
public:
    typedef typename VectorTraits<Vector>::value_type value_type;
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
    if( sqrt( nrm2r) < eps*( nrmb + nrmb_correction))  // ~
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
    cInvert()
    {   multiplyWeights_ = true;
        set_extrapolationType(2);
        nrmb_correction_ = 1.;
    }
    cInvert(const container& copyable, unsigned max_iter, value_type eps, int extrapolationType = 2, bool multiplyWeights = true, value_type nrmb_correction = 1)
    {   construct( copyable, max_iter, eps, extrapolationType, multiplyWeights, nrmb_correction);
    }
    void construct( const container& copyable, unsigned max_iter, value_type eps, int extrapolationType = 2, bool multiplyWeights = true, value_type nrmb_correction = 1.)
    {   set_size( copyable, max_iter);
        set_accuracy( eps, nrmb_correction);
        multiplyWeights_=multiplyWeights;
        set_extrapolationType( extrapolationType);
    }
    void set_size( const container& assignable, unsigned max_iterations)
    {   cheb.construct(assignable, max_iterations);
        phi0 = phi1 = phi2 = assignable;
    }
    void set_accuracy( value_type eps, value_type nrmb_correction = 1.)
    {   eps_ = eps;
        nrmb_correction_ = nrmb_correction;
    }
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
    void set_max( unsigned new_max)
    {   cheb.set_grade( new_max);
    }
    unsigned get_max() const
    {   return cheb.get_grade();
    }
    const container& get_last() const
    {   return phi0;
    }
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
