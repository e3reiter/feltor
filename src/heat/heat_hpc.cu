#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG

#include <cusp/coo_matrix.h>
#include <cusp/print.h>

#include "dg/backend/xspacelib.cuh"
#include "dg/backend/timer.cuh"
#include "dg/backend/interpolation.cuh"
// #include "dg/backend/ell_interpolation.h"
#include "file/read_input.h"
#include "file/nc_utilities.h"
#include "dg/runge_kutta.h"
#include "dg/multistep.h"
#include "dg/elliptic.h"
#include "dg/cg.h"
// #include "solovev/geometry.h"
#include "geometry_g.h"
#include "parameters.h"

#include "heat.cuh"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - writes outputs to a given outputfile using hdf5. 
        density fields are the real densities in XSPACE ( not logarithmic values)
*/

const unsigned k = 3;//!< a change in k needs a recompilation

int main( int argc, char* argv[])
{
    ////////////////////////Parameter initialisation//////////////////////////
    std::vector<double> v,v3;
    std::string input, geom;
    if(!(( argc == 4) || ( argc == 5)) )
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [geomfile] [output.nc] [input.nc]\n";
        std::cerr << "OR "<< argv[0]<<" [inputfile] [geomfile] [output.nc] \n";
        return -1;
    }
    else 
    {

        try{
            input = file::read_file( argv[1]);
            geom = file::read_file( argv[2]);
            v = file::read_input( argv[1]);
            v3 = file::read_input( argv[2]); 
        }catch( toefl::Message& m){
            m.display();
            std::cout << input << std::endl;
            std::cout << geom << std::endl;
            return -1;
        }
    }
    const eule::Parameters p( v);
    p.display( std::cout);
    const solovev::GeomParameters gp(v3);
    gp.display( std::cout);
    //////////////////////////////open nc file//////////////////////////////////
    
    ///////////////////////////////////////////////////////////////////////////
    ////////////////////////////////set up computations///////////////////////////

    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;

    //Make grids
    dg::Grid3d<double > grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n, p.Nx, p.Ny, p.Nz, p.bc, p.bc, dg::PER, dg::cylindrical);  
/*    dg::Grid3d<double > grid_out( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n_out, p.Nx_out, p.Ny_out,1.0,p.bc, p.bc, dg::PER, dg::cylindrical); */ 
    dg::Grid3d<double > grid_out( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n_out, p.Nx_out, p.Ny_out,p.Nz_out,p.bc, p.bc, dg::PER, dg::cylindrical); 
    dg::DVec w3d =  dg::create::weights(grid);
    dg::DVec w3dout =  dg::create::weights(grid_out);

    // /////////////////////get last temperature field of sim
    double normTend=0.,normTendc=1e-14;
    dg::DVec Tend(dg::evaluate(dg::zero,grid_out));
    dg::DVec Tendc(dg::evaluate(dg::zero,grid));
        dg::DVec transfer(  dg::evaluate(dg::zero, grid));
    dg::DVec transferD( dg::evaluate(dg::zero, grid_out));
    dg::HVec transferH( dg::evaluate(dg::zero, grid_out));
    dg::HVec transferHc( dg::evaluate(dg::zero, grid));
    int  tvarID;
    if (argc == 5)
    {
        file::NC_Error_Handle errin;
        int ncidin;
        errin = nc_open( argv[4], NC_NOWRITE, &ncidin);
        ///////////////////read in and show inputfile und geomfile//////////////////
        size_t length;
        errin = nc_inq_attlen( ncidin, NC_GLOBAL, "inputfile", &length);
        std::string inputin( length, 'x');
        errin = nc_get_att_text( ncidin, NC_GLOBAL, "inputfile", &inputin[0]);
        errin = nc_inq_attlen( ncidin, NC_GLOBAL, "geomfile", &length);
        std::string geomin( length, 'x');
        errin = nc_get_att_text( ncidin, NC_GLOBAL, "geomfile", &geomin[0]);
        std::cout << "input in"<<inputin<<std::endl;
        std::cout << "geome in"<<geomin <<std::endl;
        const eule::Parameters pin(file::read_input( inputin));
        const solovev::GeomParameters gpin(file::read_input( geomin));
        double Rminin=gpin.R_0-pin.boxscaleRm*gpin.a;
        double Zminin=-pin.boxscaleZm*gpin.a*gpin.elongation;
        double Rmaxin=gpin.R_0+pin.boxscaleRp*gpin.a; 
        double Zmaxin=pin.boxscaleZp*gpin.a*gpin.elongation;
        dg::Grid3d<double > grid_in( Rminin,Rmaxin, Zminin,Zmaxin, 0, 2.*M_PI, pin.n, pin.Nx, pin.Ny, pin.Nz, pin.bc, pin.bc, dg::PER, dg::cylindrical);
        size_t start3din[4]  = {pin.maxout, 0, 0, 0};
        size_t count3din[4]  = {1, grid_in.Nz(), grid_in.n()*grid_in.Ny(), grid_in.n()*grid_in.Nx()};
        std::string namesin[1] = {"T"}; 
        int dataIDsin[1]; 
        int dim_idsin[4];
        errin = nc_inq_varid(ncidin, namesin[0].data(), &dataIDsin[0]);      
        errin = nc_get_vara_double( ncidin, dataIDsin[0], start3din, count3din, transferH.data());
        Tend=(dg::DVec)transferH;
        errin = nc_close(ncidin);     
        normTend = dg::blas2::dot( w3dout, Tend);
    }
    // /////////////////////create RHS 
    std::cout << "Constructing Feltor...\n";
    eule::Feltor<dg::DMatrix, dg::DVec, dg::DVec > feltor( grid, p,gp); 
    std::cout << "Constructing Rolkar...\n";
    eule::Rolkar<dg::DMatrix, dg::DVec, dg::DVec > rolkar( grid, p,gp);
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    //initial perturbation
    dg::Gaussian3d init0(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma_z, p.amp);
//      dg::Gaussian init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, p.sigma, p.sigma, p.amp);

//     dg::BathRZ init0(16,16,p.Nz,Rmin,Zmin, 30.,5.,p.amp);
//     solovev::ZonalFlow init0(p, gp);
//     dg::CONSTANT init0( 0.);

    //background profile
    solovev::Nprofile prof(p, gp); //initial background profile
    std::vector<dg::DVec> y0(1, dg::evaluate( prof, grid)), y1(y0); 
    //field aligning
//     dg::CONSTANT gaussianZ( 1.);
//     dg::GaussianZ gaussianZ( M_PI, p.sigma_z*M_PI, 1);
//     y1[0] = feltor.dz().evaluate( init0, gaussianZ, (unsigned)p.Nz/2, 3); //rounds =2 ->2*2-1
//     y1[2] = dg::evaluate( gaussianZ, grid);
//     dg::blas1::pointwiseDot( y1[1], y1[2], y1[1]);
    //no field aligning
    y1[0] = dg::evaluate( init0, grid);
    
    dg::blas1::axpby( 1., y1[0], 1., y0[0]); //initialize ni
//     dg::blas1::transform(y0[0], y0[0], dg::PLUS<>(-1)); //initialize ni-1

//     dg::blas1::pointwiseDot(rolkar.damping(),y0[0], y0[0]); //damp with gaussprofdamp
    
    //RK solver
//     dg::RK<4, std::vector<dg::DVec> >  rk( y0);
    //SIRK solver
    dg::SIRK<std::vector<dg::DVec> > sirk(y0, grid.size(),p.eps_time);
//     dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(),1e-13);
//     karniadakis.init( feltor, rolkar, y0, p.dt);

    feltor.energies( y0);//now energies and potential are at time 0
    dg::DVec T0 = dg::evaluate( dg::one, grid);  
    dg::DVec T1 = dg::evaluate( dg::one, grid);  

    dg::blas1::axpby( 1., y0[0], 0., T0); //initialize ni
    double normT0 = dg::blas2::dot(  w3d, T0);
    double error = 0.,relerror=0.;
    /////////////////////////////set up netcdf for output/////////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    err = nc_create( argv[3],NC_NETCDF4|NC_CLOBBER, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    err = nc_put_att_text( ncid, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim_ids[4];
    err = file::define_dimensions( ncid, dim_ids, &tvarID, grid_out);
//     solovev::FieldR fieldR(gp);
//     solovev::FieldZ fieldZ(gp);
//     solovev::FieldP fieldP(gp);
//     dg::HVec vecR = dg::evaluate( fieldR, grid_out);
//     dg::HVec vecZ = dg::evaluate( fieldZ, grid_out);
//     dg::HVec vecP = dg::evaluate( fieldP, grid_out);
//     int vecID[3];
//     err = nc_def_var( ncid, "BR", NC_DOUBLE, 3, &dim_ids[1], &vecID[0]);
//     err = nc_def_var( ncid, "BZ", NC_DOUBLE, 3, &dim_ids[1], &vecID[1]);
//     err = nc_def_var( ncid, "BP", NC_DOUBLE, 3, &dim_ids[1], &vecID[2]);
//     err = nc_enddef( ncid);
//     err = nc_put_var_double( ncid, vecID[0], vecR.data());
//     err = nc_put_var_double( ncid, vecID[1], vecZ.data());
//     err = nc_put_var_double( ncid, vecID[2], vecP.data());
//     err = nc_redef(ncid);
    //field IDs
    std::string names[1] = {"T"}; 
    int dataIDs[1]; 
    err = nc_def_var( ncid, names[0].data(), NC_DOUBLE, 4, dim_ids, &dataIDs[0]);

    //energy IDs
    int EtimeID, EtimevarID;
    err = file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    int energyID, massID, energyIDs[1], dissID, dEdtID, accuracyID,errorID,relerrorID;
    err = nc_def_var( ncid, "energy",   NC_DOUBLE, 1, &EtimeID, &energyID);
    err = nc_def_var( ncid, "mass",   NC_DOUBLE, 1, &EtimeID, &massID);
    std::string energies[1] = {"Se"}; 
    err = nc_def_var( ncid, energies[0].data(), NC_DOUBLE, 1, &EtimeID, &energyIDs[0]);
    err = nc_def_var( ncid, "dissipation",   NC_DOUBLE, 1, &EtimeID, &dissID);
    err = nc_def_var( ncid, "dEdt",     NC_DOUBLE, 1, &EtimeID, &dEdtID);
    err = nc_def_var( ncid, "accuracy", NC_DOUBLE, 1, &EtimeID, &accuracyID);
    err = nc_def_var( ncid, "error", NC_DOUBLE, 1, &EtimeID, &errorID);
    err = nc_def_var( ncid, "relerror", NC_DOUBLE, 1, &EtimeID, &relerrorID);

    err = nc_enddef(ncid);
    ///////////////////////////////////first output/////////////////////////
    std::cout << "First output ... \n";
    size_t start[4] = {0, 0, 0, 0};
    size_t count[4] = {1, grid_out.Nz(), grid_out.n()*grid_out.Ny(), grid_out.n()*grid_out.Nx()};

    dg::HVec avisual( grid_out.size(), 0.);

    //interpolate coarse grid on fine grid
    dg::DMatrix interpolate = dg::create::interpolation( grid_out, grid);
//     cusp::ell_matrix<int, double, cusp::device_memory> interpolate = dg::create::ell_interpolation( grid_out, grid);
    //interpolate fine grid on coarse grid
//     dg::HMatrix interpolatec = dg::create::interpolation( grid, grid_out);
//     cusp::ell_matrix<int, double, cusp::device_memory> interpolatec = dg::create::ell_interpolation( grid, grid_out); 
    
    dg::blas2::symv( interpolate, y0[0], transferD);
    err = nc_open(argv[3], NC_WRITE, &ncid);
    transferH =transferD;
//     transferH =y0[0]; //without interp
    err = nc_put_vara_double( ncid, dataIDs[0], start, count, transferH.data());
        
        
    double time = 0;
    err = nc_put_vara_double( ncid, tvarID, start, count, &time);
    err = nc_put_vara_double( ncid, EtimevarID, start, count, &time);

    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    double energy0 = feltor.energy(), mass0 = feltor.mass(), E0 = energy0, mass = mass0, E1 = 0.0, dEdt = 0., diss = 0., accuracy=0.;
    dg::blas1::axpby( 1., y0[0], -1.,T0, T1);
    error = sqrt(dg::blas2::dot( w3d, T1)/normT0);
    if (argc==5)
    {
        //interpolate fine grid one coarse grid
//         dg::blas2::symv( interpolatec, Tend, Tendc);
//         normTendc=dg::blas2::dot(w3d,Tendc);
        Tendc=Tend;
        normTendc=dg::blas2::dot(w3d,Tendc);

//         transferHc = (dg::HVec) y0[0];
        dg::blas1::axpby( 1., y0[0], -1.,Tendc,transferD);
        relerror = sqrt(dg::blas2::dot( w3d, transferD)/normTendc);  
        //if in is finer
//         dg::blas1::axpby( 1., transferH, -1.,Tend,transferH);
//         relerror = sqrt(dg::blas2::dot( w3dout, transferH)/normTend);  
        
        
    }
    std::vector<double> evec = feltor.energy_vector();
    double Se0 = evec[0];
    double senorm = evec[0]/Se0;
    double dEdtnorm = dEdt/Se0;
    double dissnorm = diss/Se0;
    err = nc_put_vara_double( ncid, energyID, Estart, Ecount, &energy0);
    err = nc_put_vara_double( ncid, massID,   Estart, Ecount, &mass0);
    err = nc_put_vara_double( ncid, energyIDs[0], Estart, Ecount, &senorm);
    err = nc_put_vara_double( ncid, dissID,     Estart, Ecount,&dissnorm);
    err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount,&dEdtnorm);
    err = nc_put_vara_double( ncid, accuracyID, Estart, Ecount,&accuracy);
    err = nc_put_vara_double( ncid, errorID, Estart, Ecount,&error);
    err = nc_put_vara_double( ncid, relerrorID, Estart, Ecount,&relerror);

    err = nc_close(ncid);
    std::cout << "First write successful!\n";

    ///////////////////////////////////////Timeloop/////////////////////////////////
    dg::Timer t;
    t.tic();
#ifdef DG_BENCHMARK
    unsigned step = 0;
#endif //DG_BENCHMARK
    for( unsigned i=1; i<=p.maxout; i++)
    {

#ifdef DG_BENCHMARK
        dg::Timer ti;
        ti.tic();
#endif//DG_BENCHMARK
        for( unsigned j=0; j<p.itstp; j++)
        {
            try{
//                 rk( feltor, y0, y1, p.dt); //RK stepper
                sirk(feltor,rolkar,y0,y1,p.dt); //SIRK stepper
//                 karniadakis( feltor, rolkar, y0);  //Karniadakis stepper
                y0.swap( y1);}
              catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                err = nc_close(ncid);
                return -1;}
            step++;
            time+=p.dt;
            feltor.energies(y0);//advance potential and energies
            Estart[0] = step;
            E1 = feltor.energy(), mass = feltor.mass(), diss = feltor.energy_diffusion();
            dEdt = (E1 - E0)/p.dt; 
            E0 = E1;
            dg::blas1::axpby( 1., y0[0], -1.,T0, T1);
            error = sqrt(dg::blas2::dot( w3d, T1)/normT0);
            if (argc==5)
            {
//                 dg::blas2::symv( interpolate, y0[0], transferD);
//                 transferH =transferD;
//                 dg::blas1::axpby( 1., transferH, -1.,Tend, transferH);
//                 relerror = sqrt(dg::blas2::dot( w3dout, transferH)/normTend);
//                         dg::blas2::symv( interpolatec, Tend, Tendc);
//         normTendc=dg::blas2::dot(w3d,Tendc);
//                 dg::blas2::symv( interpolatec, Tend, Tendc);
//                 normTendc=dg::blas2::dot(w3d,Tendc);
                Tendc=Tend;
                normTendc=dg::blas2::dot(w3d,Tendc);

//                 transferHc =(dg::HVec) y0[0];
                dg::blas1::axpby( 1., y0[0], -1.,Tendc,transferD);
                relerror = sqrt(dg::blas2::dot( w3d, transferD)/normTendc);  
            }
            accuracy = 2.*fabs( (dEdt-diss)/(dEdt + diss));
            evec = feltor.energy_vector();
            senorm = evec[0]/Se0;
            dEdtnorm = dEdt/Se0;
            dissnorm = diss/Se0;
            err = nc_open(argv[3], NC_WRITE, &ncid);
            err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
            err = nc_put_vara_double( ncid, energyID, Estart, Ecount, &E1);
            err = nc_put_vara_double( ncid, massID,   Estart, Ecount, &mass);
            err = nc_put_vara_double( ncid, energyIDs[0], Estart, Ecount, &senorm);
            err = nc_put_vara_double( ncid, dissID,     Estart, Ecount,&dissnorm);
            err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount,&dEdtnorm);
            err = nc_put_vara_double( ncid, accuracyID, Estart, Ecount,&accuracy);
            err = nc_put_vara_double( ncid, errorID, Estart, Ecount,&error);
            err = nc_put_vara_double( ncid, relerrorID, Estart, Ecount,&relerror);

            std::cout << "(m_tot-m_0)/m_0: "<< (feltor.mass()-mass0)/mass0<<"\t";
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout <<" d E/dt = " << dEdt <<" Lambda = " << diss << " -> Accuracy: "<< accuracy << " -> Error: "<< error <<" -> Error2: "<< relerror <<"\n";
            err = nc_close(ncid);

        }
#ifdef DG_BENCHMARK
        ti.toc();
        std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
#endif//DG_BENCHMARK
        //////////////////////////write fields////////////////////////
        start[0] = i;

        dg::blas2::symv( interpolate, y0[0], transferD);
        transferH =transferD;
//         transferH =y0[0];
        err = nc_open(argv[3], NC_WRITE, &ncid);
//         transferH = avisual;
        err = nc_put_vara_double( ncid, dataIDs[0], start, count, transferH.data());
        
        err = nc_put_vara_double( ncid, tvarID, start, count, &time);
        err = nc_close(ncid);
    }
    t.toc(); 
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout<<"s/step\n";

    return 0;

}
