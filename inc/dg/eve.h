#ifndef _DG_EVE_
#define _DG_EVE_

#include <cmath>

#include "blas.h"
#include "functors.h"

#ifdef DG_BENCHMARK
#include "backend/timer.cuh"
#endif //DG_BENCHMARK

namespace dg
{

template< class Vector>
class EVE
{
public:
    typedef typename VectorTraits<Vector>::value_type value_type;//!< value type of the Vector class
    EVE() {}
    EVE( const Vector& copyable, unsigned max_iter):r( copyable), p( r), ap( r), max_iter( max_iter) {}
    void set_max( unsigned new_max)
    {   max_iter = new_max;
    }
    unsigned get_max() const
    {   return max_iter;
    }
    void construct( const Vector& copyable, unsigned max_iterations)
    {   ap = p = r = copyable;
        max_iter = max_iterations;
    }
    template< class Matrix>
    unsigned operator()( Matrix& A, Vector& x, const Vector& b, value_type& ev_max, value_type eps=1e-12, value_type nrmb_correction = 1, value_type eps_ev=1e-16);
private:
    Vector r, p, ap;
    unsigned max_iter;
};

template< class Vector>
template< class Matrix>
unsigned EVE< Vector>::operator()( Matrix& A, Vector& x, const Vector& b, value_type& ev_max, value_type eps, value_type nrmb_correction, value_type eps_ev)
{   value_type nrmb = sqrt( blas1::dot( b, b));
#ifdef DG_DEBUG
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( rank==0)
#endif //MPI
    {   std::cout << "Norm of b "<<nrmb <<"\n";
        std::cout << "Residual errors: \n";
    }
#endif //DG_DEBUG
    blas2::symv( A, x, r);
    blas1::axpby( 1., b, -1., r);
    value_type nrm2r_old = blas1::dot( r,r);
    if( sqrt( nrm2r_old ) < eps*(nrmb + nrmb_correction)) // ignore as only called once?
    {   return 0;                                         // ?
    }                                                     // ?
    blas1::axpby(1., r, 0., p);
    value_type nrm2r_new, nrmAp;
    value_type alpha = 1., alpha_inv = 1., delta = 0.;
    value_type evdash, gamma = 0., lambda, omega, beta = 0.;
    value_type ev_est = 0.;
    for( unsigned i=1; i<max_iter; i++)
    {   lambda = delta*alpha_inv;       // EVE!
        blas2::symv( A, p, ap);
        nrmAp = blas1::dot( p, ap);
        alpha = nrm2r_old /nrmAp;
        alpha_inv = nrmAp /nrm2r_old;   // EVE!
        lambda += alpha_inv;            // EVE!
        blas1::axpby( alpha, p, 1., x);
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
        delta = nrm2r_new /nrm2r_old;                  // EVE!
        evdash = ev_est -lambda;                       // EVE!
        omega = sqrt( evdash*evdash +4.*beta*gamma);   // EVE!
        gamma = 0.5 *(1. -evdash /omega);              // EVE!
        ev_max += omega*gamma;                         // EVE!
        //        if( abs(ev_est-ev_max) < eps_ev) // ?
        //        {   return i;                    // ?
        //        }                                // ?
        beta = delta*alpha_inv*alpha_inv;              // EVE!
        if( sqrt( nrm2r_new) < eps*(nrmb + nrmb_correction)) // ?
        {   return i;                                        // ?
        }                                                    // ?
        blas1::axpby(1., r, delta, p);
        nrm2r_old=nrm2r_new;
        ev_est = ev_max;                                     // ?
    }
    return max_iter;
};

template<class container>
struct eInvert
{   typedef typename VectorTraits<container>::value_type value_type;
    eInvert()
    {   multiplyWeights_ = true;
        set_extrapolationType(2);
        nrmb_correction_ = 1.;
    }
    eInvert(const container& copyable, unsigned max_iter, value_type eps, int extrapolationType = 2, bool multiplyWeights = true, value_type nrmb_correction = 1)
    {   construct( copyable, max_iter, eps, extrapolationType, multiplyWeights, nrmb_correction);
    }
    void construct( const container& copyable, unsigned max_iter, value_type eps, int extrapolationType = 2, bool multiplyWeights = true, value_type nrmb_correction = 1.)
    {   set_size( copyable, max_iter);
        set_accuracy( eps, nrmb_correction);
        multiplyWeights_=multiplyWeights;
        set_extrapolationType( extrapolationType);
    }
    void set_size( const container& assignable, unsigned max_iterations)
    {   cg.construct(assignable, max_iterations);
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
    {   cg.set_max( new_max);
    }
    unsigned get_max() const
    {   return cg.get_max();
    }
    const container& get_last() const
    {   return phi0;
    }


    template< class SymmetricOp, class Preconditioner >
    unsigned operator()( SymmetricOp& op, container& phi, const container& rho, const container& w, Preconditioner& p, const container& inv_weights, value_type& ev_max)
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
            number = cg( op, phi, phi2, ev_max, eps_, nrmb_correction_);
        }
        else
            number = cg( op, phi, rho, ev_max, eps_, nrmb_correction_);
#ifdef DG_BENCHMARK
#ifdef MPI_VERSION
        if(rank==0)
#endif //MPI
            std::cout << "# of cg iterations \t"<< number << "\t";
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
    dg::EVE< container > cg;
    value_type alpha[3];
    bool multiplyWeights_;
};

} //namespace dg
#endif //_DG_EVE_
