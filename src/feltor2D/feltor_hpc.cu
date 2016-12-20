#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>
// #define DG_DEBUG



#include "dg/backend/xspacelib.cuh"
#include "dg/backend/timer.cuh"
#include "dg/backend/interpolation.cuh"
#include "file/read_input.h"
#include "file/nc_utilities.h"
#include "geometries/solovev.h"

#include "feltor/feltor.cuh"
#include "feltor/parameters.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - writes outputs to a given outputfile using hdf5. 
        density fields are the real densities in XSPACE ( not logarithmic values)
*/

typedef dg::FieldAligned< dg::CylindricalGrid3d<dg::DVec>, dg::IDMatrix, dg::DVec> DFA;
int main( int argc, char* argv[])
{
    ////////////////////////Parameter initialisation//////////////////////////
    std::vector<double> v,v3;
    std::string input, geom;
    if( argc != 4)
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [geomfile] [outputfile]\n";
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
    ////////////////////////////////set up computations///////////////////////////

    double Rmin=gp.R_0-p.boxscaleRm*gp.a;
    double Zmin=-p.boxscaleZm*gp.a*gp.elongation;
    double Rmax=gp.R_0+p.boxscaleRp*gp.a; 
    double Zmax=p.boxscaleZp*gp.a*gp.elongation;
    //Make grids
    dg::CylindricalGrid3d<dg::DVec> grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n, p.Nx, p.Ny, 1, p.bc, p.bc, dg::PER);  
    dg::CylindricalGrid3d<dg::DVec> grid_out( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n_out, p.Nx_out, p.Ny_out,1,p.bc, p.bc, dg::PER);  
    //create RHS 
    std::cout << "Constructing Feltor...\n";
    eule::Feltor<dg::CylindricalGrid3d<dg::DVec>, dg::DS<DFA, dg::DMatrix, dg::DVec>, dg::DMatrix, dg::DVec > feltor( grid, p, gp); //initialize before rolkar!
    std::cout << "Constructing Rolkar...\n";
    eule::Rolkar<dg::CylindricalGrid3d<dg::DVec>, dg::DS<DFA, dg::DMatrix, dg::DVec>, dg::DMatrix, dg::DVec> rolkar( grid, p, gp, feltor.ds(), feltor.dsDIR());
    std::cout << "Done!\n";

    /////////////////////The initial field///////////////////////////////////////////
    //background profile
    solovev::Nprofile prof(p.bgprofamp, p.nprofileamp, gp); //initial background profile
    std::vector<dg::DVec> y0(4, dg::evaluate( prof, grid)), y1(y0); 
    //initial perturbation
    if (p.mode == 0  || p.mode ==1) 
    { 
        dg::Gaussian3d init0( gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI, p.sigma, p.sigma, p.sigma, p.amp);
        y1[1] = dg::evaluate( init0, grid);
    }
    if (p.mode == 2) 
    { 
        dg::BathRZ init0(16,16,1,Rmin,Zmin, 30.,5.,p.amp);
        y1[1] = dg::evaluate( init0, grid);
    }
    if (p.mode == 3) 
    { 
        solovev::ZonalFlow init0(p.amp, p.k_psi, gp);
        y1[1] = dg::evaluate( init0, grid);
    }
    
    dg::blas1::axpby( 1., y1[1], 1., y0[1]); //initialize ni
    dg::blas1::transform(y0[1], y0[1], dg::PLUS<>(-1)); //initialize ni-1
    dg::DVec damping = dg::evaluate( solovev::GaussianProfXDamping( gp), grid);
    dg::blas1::pointwiseDot( damping, y0[1], y0[1]); //damp with gaussprofdamp
    std::cout << "initialize ne" << std::endl;
    feltor.initializene( y0[1], y0[0]);    
    std::cout << "Done!\n";

    dg::blas1::axpby( 0., y0[2], 0., y0[2]); //set Ue = 0
    dg::blas1::axpby( 0., y0[3], 0., y0[3]); //set Ui = 0
    
    std::cout << "initialize karniadakis" << std::endl;
    dg::Karniadakis< std::vector<dg::DVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    karniadakis.init( feltor, rolkar, y0, p.dt);
//     feltor.energies(y0); //now energies and potential are at time 0
    std::cout << "Done!\n";
    /////////////////////////////set up netcdf/////////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    err = nc_create( argv[3],NC_NETCDF4|NC_CLOBBER, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    err = nc_put_att_text( ncid, NC_GLOBAL, "geomfile", geom.size(), geom.data());
    int dim_ids[4], tvarID;
    err = file::define_dimensions( ncid, dim_ids, &tvarID, grid_out);
    solovev::FieldR fieldR(gp);
    solovev::FieldZ fieldZ(gp);
    solovev::FieldP fieldP(gp);
    dg::HVec vecR = dg::evaluate( fieldR, grid_out);
    dg::HVec vecZ = dg::evaluate( fieldZ, grid_out);
    dg::HVec vecP = dg::evaluate( fieldP, grid_out);
    int vecID[3];
    err = nc_def_var( ncid, "BR", NC_DOUBLE, 3, &dim_ids[1], &vecID[0]);
    err = nc_def_var( ncid, "BZ", NC_DOUBLE, 3, &dim_ids[1], &vecID[1]);
    err = nc_def_var( ncid, "BP", NC_DOUBLE, 3, &dim_ids[1], &vecID[2]);
    err = nc_enddef( ncid);
    err = nc_put_var_double( ncid, vecID[0], vecR.data());
    err = nc_put_var_double( ncid, vecID[1], vecZ.data());
    err = nc_put_var_double( ncid, vecID[2], vecP.data());
    err = nc_redef(ncid);

    //field IDs
    std::string names[5] = {"electrons", "ions", "Ue", "Ui", "potential"}; 
    int dataIDs[5]; 
    for( unsigned i=0; i<5; i++){
        err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 4, dim_ids, &dataIDs[i]);}

    //energy IDs
    int EtimeID, EtimevarID;
    err = file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    int energyID, massID, energyIDs[5], dissID, dEdtID, accuracyID;
    err = nc_def_var( ncid, "energy",   NC_DOUBLE, 1, &EtimeID, &energyID);
    err = nc_def_var( ncid, "mass",   NC_DOUBLE, 1, &EtimeID, &massID);
    std::string energies[5] = {"Se", "Si", "Uperp", "Upare", "Upari"}; 
    for( unsigned i=0; i<5; i++){
        err = nc_def_var( ncid, energies[i].data(), NC_DOUBLE, 1, &EtimeID, &energyIDs[i]);}
    err = nc_def_var( ncid, "dissipation",   NC_DOUBLE, 1, &EtimeID, &dissID);
    err = nc_def_var( ncid, "dEdt",     NC_DOUBLE, 1, &EtimeID, &dEdtID);
    err = nc_def_var( ncid, "accuracy", NC_DOUBLE, 1, &EtimeID, &accuracyID);
    err = nc_enddef(ncid);
    ///////////////////////////////////first output/////////////////////////
    std::cout << "First output ... \n";
    size_t start[4] = {0, 0, 0, 0};
    size_t count[4] = {1, grid_out.Nz(), grid_out.n()*grid_out.Ny(), grid_out.n()*grid_out.Nx()};
    dg::DVec transfer(  dg::evaluate(dg::zero, grid));
    dg::DVec transferD( dg::evaluate(dg::zero, grid_out));
    dg::HVec transferH( dg::evaluate(dg::zero, grid_out));
    dg::IDMatrix interpolate = dg::create::interpolation( grid_out, grid); 
    for( unsigned i=0; i<4; i++)
    {
        dg::blas2::symv( interpolate, y0[i], transferD);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[i], start, count, transferH.data() );
    }
    transfer = feltor.potential()[0];
    dg::blas2::symv( interpolate, transfer, transferD);
    dg::blas1::transfer( transferD, transferH);
    err = nc_put_vara_double( ncid, dataIDs[4], start, count, transferH.data() );
    double time = 0;
    err = nc_put_vara_double( ncid, tvarID, start, count, &time);
    err = nc_put_vara_double( ncid, EtimevarID, start, count, &time);

    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    double energy0 = feltor.energy(), mass0 = feltor.mass(), E0 = energy0, mass = mass0, E1 = 0.0, dEdt = 0., diss = 0., accuracy=0.;
    std::vector<double> evec = feltor.energy_vector();
    err = nc_put_vara_double( ncid, energyID, Estart, Ecount, &energy0);
    err = nc_put_vara_double( ncid, massID,   Estart, Ecount, &mass0);
    for( unsigned i=0; i<5; i++)
        err = nc_put_vara_double( ncid, energyIDs[i], Estart, Ecount, &evec[i]);

    err = nc_put_vara_double( ncid, dissID,     Estart, Ecount,&diss);
    err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount,&dEdt);
    err = nc_put_vara_double( ncid, accuracyID, Estart, Ecount,&accuracy);
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
            try{ karniadakis( feltor, rolkar, y0);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                err = nc_close(ncid);
                return -1;
            }
            step++;
            time+=p.dt;
//             feltor.energies(y0);//advance potential and energies
            Estart[0] = step;
            E1 = feltor.energy(), mass = feltor.mass(), diss = feltor.energy_diffusion();
            dEdt = (E1 - E0)/p.dt; 
            E0 = E1;
            accuracy = 2.*fabs( (dEdt-diss)/(dEdt + diss));
            evec = feltor.energy_vector();
            err = nc_open(argv[3], NC_WRITE, &ncid);
            err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
            err = nc_put_vara_double( ncid, energyID, Estart, Ecount, &E1);
            err = nc_put_vara_double( ncid, massID,   Estart, Ecount, &mass);
            for( unsigned i=0; i<5; i++)
            {
                err = nc_put_vara_double( ncid, energyIDs[i], Estart, Ecount, &evec[i]);
            }
            err = nc_put_vara_double( ncid, dissID,     Estart, Ecount,&diss);
            err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount,&dEdt);
            err = nc_put_vara_double( ncid, accuracyID, Estart, Ecount,&accuracy);
            std::cout << "(m_tot-m_0)/m_0: "<< (feltor.mass()-mass0)/mass0<<"\t";
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout <<" d E/dt = " << dEdt <<" Lambda = " << diss << " -> Accuracy: "<< accuracy << "\n";
            err = nc_close(ncid);

        }
#ifdef DG_BENCHMARK
        ti.toc();
        std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
#endif//DG_BENCHMARK
        //////////////////////////write fields////////////////////////
        start[0] = i;
        err = nc_open(argv[3], NC_WRITE, &ncid);
        for( unsigned j=0; j<4; j++)
        {
            dg::blas2::symv( interpolate, y0[j], transferD);
            dg::blas1::transfer( transferD, transferH);
            err = nc_put_vara_double( ncid, dataIDs[j], start, count, transferH.data());
        }
        transfer = feltor.potential()[0];
        dg::blas2::symv( interpolate, transfer, transferD);
        dg::blas1::transfer( transferD, transferH);
        err = nc_put_vara_double( ncid, dataIDs[4], start, count, transferH.data() );
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

