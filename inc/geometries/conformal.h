#pragma once

#include "hector.h"



namespace dg{
namespace conformal
{

typedef dg::Hector<dg::IHMatrix, dg::HMatrix, dg::HVec> Generator;

///@cond
template< class container>
struct RingGrid2d; 
///@endcond

/**
 * @brief A three-dimensional grid based on conformal coordintes generated by hector
 *
 * @tparam container Vector class that holds metric coefficients
 */
template< class container>
struct RingGrid3d : public dg::Grid3d<double>
{
    typedef dg::ConformalCylindricalTag metric_category; //!< metric tag
    typedef RingGrid2d<container> perpendicular_grid; //!< the two-dimensional grid type

    /**
     * @brief Construct 
     *
     * @param gp The geometric parameters define the magnetic field
     * @param psi_0 lower boundary for psi
     * @param psi_1 upper boundary for psi
     * @param n The dG number of polynomials
     * @param Nx The number of points in x-direction
     * @param Ny The number of points in y-direction
     * @param Nz The number of points in z-direction
     * @param bcx The boundary condition in x (y,z are periodic)
     */
    RingGrid3d( solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx): 
        dg::Grid3d<double>( 0, 1, 0., 2.*M_PI, 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER)
    { 
        assert( psi_1 != psi_0);
        solovev::Psip psip( gp); 
        solovev::PsipR psipR( gp); 
        solovev::PsipZ psipZ( gp); 
        solovev::LaplacePsip lap( gp); 
        dg::Hector<dg::IHMatrix, dg::HMatrix, dg::HVec> hector( psip, psipR, psipZ, lap, psi_0, psi_1, gp.R_0, 0.);

        construct( hector, n, Nx, Ny, Nz, bcx, dg::ConformalTag());
    }

    template< class Generator>
    RingGrid3d( const Generator& hector, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx) :
        dg::Grid3d<double>( 0, 1, 0., 2.*M_PI, 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER)
    {
        construct( hector, n,Nx, Ny, Nz, bcx, typename dg::GeometryTraits<Generator>::metric_category());
    }
    perpendicular_grid perp_grid() const { return conformal::RingGrid2d<container>(*this);}

    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const container& g_xx()const{return gradU2_;}
    const container& g_yy()const{return gradU2_;}
    const container& g_pp()const{return g_pp_;}
    const container& vol()const{return vol_;}
    const container& perpVol()const{return vol2d_;}
    private:
    template< class Generator>
    void construct( const Generator& hector, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, dg::ConformalTag ) 
    {
        dg::Grid2d<double> guv( 0., hector.lu(), 0., 2.*M_PI, n, Nx, Ny );
        dg::Grid1d<double> gu( 0., hector.lu(), n, Nx);
        dg::Grid1d<double> gv( 0., 2.*M_PI, n, Ny);
        const dg::HVec u1d = dg::evaluate( dg::coo1, gu);
        const dg::HVec v1d = dg::evaluate( dg::coo1, gv);
        hector( u1d, v1d, r_, z_, xr_, xz_, yr_, yz_);
        init_X_boundaries( 0., hector.lu());
        lift3d( ); //lift to 3D grid
        construct_metric();
    }
    void lift3d( )
    {
        //lift to 3D grid
        unsigned Nx = this->n()*this->Nx(), Ny = this->n()*this->Ny();
        for( unsigned k=1; k<this->Nz(); k++)
            for( unsigned i=0; i<Nx*Ny; i++)
            {
                r_[k*Nx*Ny+i] = r_[(k-1)*Nx*Ny+i];
                z_[k*Nx*Ny+i] = z_[(k-1)*Nx*Ny+i];
                xr_[k*Nx*Ny+i] = xr_[(k-1)*Nx*Ny+i];
                xz_[k*Nx*Ny+i] = xz_[(k-1)*Nx*Ny+i];
                yr_[k*Nx*Ny+i] = yr_[(k-1)*Nx*Ny+i];
                yz_[k*Nx*Ny+i] = yz_[(k-1)*Nx*Ny+i];
            }
    }
    //compute metric elements from xr, xz, yr, yz, r and z
    void construct_metric( )
    {
        thrust::host_vector<double> tempxx( r_), tempvol(r_);
        for( unsigned i = 0; i<this->size(); i++)
        {
            tempxx[i] = (xr_[i]*xr_[i]+xz_[i]*xz_[i]);
            tempvol[i] = r_[i]/ tempxx[i];
        }
        dg::blas1::transfer( tempxx, gradU2_);
        dg::blas1::transfer( tempvol, vol_);
        dg::blas1::pointwiseDivide( tempvol, r_, tempvol);
        dg::blas1::transfer( tempvol, vol2d_);
        thrust::host_vector<double> ones = dg::evaluate( dg::one, *this);
        dg::blas1::pointwiseDivide( ones, r_, tempxx);
        dg::blas1::pointwiseDivide( tempxx, r_, tempxx); //1/R^2
        g_pp_=tempxx;
    }
    
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_; //3d vector
    container gradU2_, g_pp_, vol_, vol2d_;

};

/**
 * @brief A two-dimensional grid based on the conformal coordinates generated by hector
 */
template< class container>
struct RingGrid2d : public dg::Grid2d<double>
{
    typedef dg::ConformalTag metric_category;
    template< class Generator>
    RingGrid2d( const Generator& hector, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx):
        dg::Grid2d<double>( 0, 1., 0., 2*M_PI, n,Nx,Ny, bcx, dg::PER)
    {
        conformal::RingGrid3d<container> g( hector, n,Nx,Ny,1,bcx);
        init_X_boundaries( g.x0(), g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        gradU2_=g.g_xx();
        vol2d_=g.perpVol();
    }
    RingGrid2d( const solovev::GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx): 
        dg::Grid2d<double>( 0, 1., 0., 2*M_PI, n,Nx,Ny, bcx, dg::PER)
    {
        conformal::RingGrid3d<container> g( gp, psi_0, psi_1, n,Nx,Ny,1,bcx);
        init_X_boundaries( g.x0(), g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        gradU2_=g.g_xx();
        vol2d_=g.perpVol();
    }
    RingGrid2d( const RingGrid3d<container>& g):
        dg::Grid2d<double>( g.x0(), g.x1(), g.y0(), g.y1(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy())
    {
        unsigned s = this->size();
        r_.resize( s), z_.resize(s), xr_.resize(s), xz_.resize(s), yr_.resize(s), yz_.resize(s);
        gradU2_.resize( s), vol2d_.resize(s);
        for( unsigned i=0; i<s; i++)
        { r_[i]=g.r()[i], z_[i]=g.z()[i], xr_[i]=g.xr()[i], xz_[i]=g.xz()[i], yr_[i]=g.yr()[i], yz_[i]=g.yz()[i];}
        thrust::copy( g.g_xx().begin(), g.g_xx().begin()+s, gradU2_.begin());
        thrust::copy( g.perpVol().begin(), g.perpVol().begin()+s, vol2d_.begin());
    }


    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const container& g_xx()const{return gradU2_;}
    const container& g_yy()const{return gradU2_;}
    const container& vol()const{return vol2d_;}
    const container& perpVol()const{return vol2d_;}
    private:
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_;
    container gradU2_, vol2d_;
};


}//namespace conformal
}//namespace
