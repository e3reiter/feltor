#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include "dg/blas.h"

#include "dg/backend/functions.h"
#include "geom_parameters.h"


/*!@file
 *
 * Geometry objects 
 */
namespace solovev
{
///@addtogroup geom
///@{

/**
 * @brief \f[ \hat{\psi}_p  \f]
 *
 * \f[ \hat{\psi}_p(R,Z) = 
      \hat{R}_0\Bigg\{\bar{R}^4/8 + A \left[ 1/2 \bar{R}^2  \ln{(\bar{R}   )}-(\bar{R}^4 )/8\right]
      + \sum_{i=1}^{12} c_i\bar \psi_{pi}(\bar R, \bar Z) \Bigg\}
      =
      \hat{R}_0\Bigg\{A \left[ 1/2 \bar{R}^2  \ln{(\bar{R}   )}-(\bar{R}^4 )/8\right]
      +c_1+
      c_2 \bar{R}^2 +
      c_3 \left[  \bar{Z}^2-\bar{R}^2  \ln{(\bar{R}   )} \right] +
      c_4 \left[\bar{R}^4 -4 \bar{R}^2  \bar{Z}^2 \right] +
      c_5 \left[3 \bar{R}^4  \ln{(\bar{R}   )}-9 \bar{R}^2  \bar{Z}^2-12 \bar{R}^2  \bar{Z}^2
      \ln{(\bar{R}   )}+2  \bar{Z}^4\right]+
      c_6 \left[ \bar{R}^6 -12 \bar{R}^4  \bar{Z}^2+8 \bar{R}^2  \bar{Z}^4 \right]  +
      c_7 \left[-(15 \bar{R}^6  \ln{(\bar{R}   )})+75 \bar{R}^4  \bar{Z}^2+180 \bar{R}^4 
       \bar{Z}^2 \ln{(\bar{R}   )}-140 \bar{R}^2  \bar{Z}^4-120 \bar{R}^2  \bar{Z}^4 
      \ln{(\bar{R}   )}+8  \bar{Z}^6 \right] +
      c_8  \bar{Z}+c_9 \bar{R}^2  \bar{Z}+(\bar{R}^4 )/8 + 
      c_{10} \left[ \bar{Z}^3-3 \bar{R}^2  \bar{Z} \ln{(\bar{R}   )}\right]+ 
      c_{11} \left[3 \bar{R}^4  \bar{Z}-4 \bar{R}^2  \bar{Z}^3\right]      +
      c_{12} \left[-(45 \bar{R}^4  \bar{Z})+60 \bar{R}^4  \bar{Z} \ln{(\bar{R}   )}-
      80 \bar{R}^2  \bar{Z}^3 \ln{(\bar{R}   )}+8  \bar{Z}^5 \right] 
      \Bigg\} \f]
      with \f$ \bar R := \frac{ R}{R_0} \f$ and \f$\bar Z := \frac{Z}{R_0}\f$
 *
 */    
struct Psip
{
    /**
     * @brief Construct from given geometric parameters
     *
     * @param gp useful geometric parameters
     */
    Psip( GeomParameters gp): R_0_(gp.R_0), A_(gp.A), c_(gp.c), psi_0(gp.psipmaxcut), alpha_( gp.alpha) {}
/**
 * @brief \f$ \hat \psi_p(R,Z) \f$

      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
      @return \f$ \hat \psi_p(R,Z) \f$
 */
    double operator()(double R, double Z) const
    {    
        return psi_alt( R, Z);
    }
    /**
     * @brief \f$ \psi_p(R,Z,\phi) \equiv \psi_p(R,Z)\f$
     *
      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
      @param phi angle (cylindrical coordinates)
     *
     * @return \f$ \hat \psi_p(R,Z,\phi) \f$
     */
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
    /**
     * @brief Show parameters to std::cout
     */
    void display() const
    {
      std::cout << R_0_ <<"  " <<A_ <<"\n";
      std::cout << c_[0] <<"\n";
    }
  private:
    double psi_alt(double R, double Z) const
    {
       double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,Zn6,lgRn;
       Rn = R/R_0_; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2;
       Zn = Z/R_0_; Zn2 = Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2; Zn6 = Zn3*Zn3;
       lgRn= log(Rn);
       return   R_0_*( c_[12]*Rn4/8.+ A_ * ( 1./2.* Rn2* lgRn-(Rn4)/8.)  //c_[12] is to make fieldlines straight
                      + c_[0]  //c_[0] entspricht c_1
              + c_[1]  *Rn2
              + c_[2]  *(Zn2 - Rn2 * lgRn ) 
              + c_[3]  *(Rn4 - 4.* Rn2*Zn2 ) 
              + c_[4]  *(3.* Rn4 * lgRn  -9.*Rn2*Zn2 -12.* Rn2*Zn2 * lgRn + 2.*Zn4)
              + c_[5]  *(Rn4*Rn2-12.* Rn4*Zn2 +8.* Rn2 *Zn4 ) 
              + c_[6]  *(-15.*Rn4*Rn2 * lgRn + 75.* Rn4 *Zn2 + 180.* Rn4*Zn2 * lgRn 
                         -140.*Rn2*Zn4 - 120.* Rn2*Zn4 *lgRn + 8.* Zn6 )
              + c_[7]  *Zn
              + c_[8]  *Rn2*Zn            
                      + c_[9] *(Zn2*Zn - 3.* Rn2*Zn * lgRn)
              + c_[10] *( 3. * Rn4*Zn - 4. * Rn2*Zn3)
              + c_[11] *(-45.* Rn4*Zn + 60.* Rn4*Zn* lgRn - 80.* Rn2*Zn3* lgRn + 8. * Zn5)            
                      );
    }
    double R_0_, A_;
    std::vector<double> c_;
    double psi_0;
    double alpha_;
};
/**
 * @brief \f[ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}} \f]
 *
 * \f[ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}} =
      \Bigg\{ 2 c_2 \bar{R} +(\bar{R}^3 )/2+2 c_9 \bar{R}  \bar{Z}
      +c_4 (4 \bar{R}^3 -8 \bar{R}  \bar{Z}^2)+c_{11} 
      (12 \bar{R}^3  \bar{Z}-8 \bar{R}  \bar{Z}^3 
      +c_6 (6 \bar{R}^5 -48 \bar{R}^3  \bar{Z}^2+16 \bar{R}  \bar{Z}^4)+c_3 (-\bar{R} -2 \bar{R}  \ln{(\bar{R}   )})+
      A ((\bar{R} )/2-(\bar{R}^3 )/2+\bar{R}  \ln{(\bar{R}   )})
      +c_{10} (-3 \bar{R}  \bar{Z}-6 \bar{R}  \bar{Z} \ln{(\bar{R}   )})+c_5 (3 \bar{R}^3 -30 \bar{R}  
      \bar{Z}^2+12 \bar{R}^3  \ln{(\bar{R}   )}-24 \bar{R}  \bar{Z}^2 \ln{(\bar{R}   )}) 
      +c_{12} (-120 \bar{R}^3  \bar{Z}-80 \bar{R}  \bar{Z}^3+240 \bar{R}^3  \bar{Z} 
      \ln{(\bar{R}   )}-160 \bar{R}  \bar{Z}^3 \ln{(\bar{R}   )})
      +c_7 (-15 \bar{R}^5 +480 \bar{R}^3  \bar{Z}^2-400 \bar{R}  \bar{Z}^4-90 \bar{R}^5  
      \ln{(\bar{R}   )}+720 \bar{R}^3  \bar{Z}^2 \ln{(\bar{R}   )}-240 \bar{R}  \bar{Z}^4
      \ln{(\bar{R}   )})\Bigg\} \f]
      with \f$ \bar R := \frac{ R}{R_0} \f$ and \f$\bar Z := \frac{Z}{R_0}\f$
 */ 
struct PsipR
{
    /**
     * @brief Construct from given geometric parameters
     *
     * @param gp useful geometric parameters
     */
    PsipR( GeomParameters gp): R_0_(gp.R_0), A_(gp.A), c_(gp.c), psip_(gp) {}
/**
 * @brief \f$ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}}(R,Z)  \f$

      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
    * @return \f$ \frac{\partial  \hat{\psi}_p}{ \partial \hat{R}}(R,Z)  \f$
 */ 
    double operator()(double R, double Z) const
    {    
        return psipR_alt( R, Z);
    }
    /**
     * @brief \f$ \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}}(R,Z,\phi) \equiv \frac{\partial  \hat{\psi}_p }{ \partial \hat{R}}(R,Z)\f$
      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
      @param phi angle (cylindrical coordinates)
    * @return \f$ \frac{\partial  \hat{\psi}_p}{ \partial \hat{R}}(R,Z,\phi)  \f$
 */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }


    /**
     * @brief Print parameters to std::cout
     */
    void display() const
    {
      std::cout << R_0_ <<"  " <<A_ <<"\n";
      std::cout << c_[0] <<"\n";
    }
  private:
    double psipR_alt(double R, double Z) const
    {    
        double Rn,Rn2,Rn3,Rn5,Zn,Zn2,Zn3,Zn4,lgRn;
        Rn = R/R_0_; Rn2 = Rn*Rn; Rn3 = Rn2*Rn;  Rn5 = Rn3*Rn2; 
        Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; 
        lgRn= log(Rn);
        return   (Rn3/2.*c_[12] + (Rn/2. - Rn3/2. + Rn*lgRn)* A_ + 
        2.* Rn* c_[1] + (-Rn - 2.* Rn*lgRn)* c_[2] + (4.*Rn3 - 8.* Rn *Zn2)* c_[3] + 
        (3. *Rn3 - 30.* Rn *Zn2 + 12. *Rn3*lgRn -  24.* Rn *Zn2*lgRn)* c_[4]
        + (6 *Rn5 - 48 *Rn3 *Zn2 + 16.* Rn *Zn4)*c_[5]
        + (-15. *Rn5 + 480. *Rn3 *Zn2 - 400.* Rn *Zn4 - 90. *Rn5*lgRn + 
            720. *Rn3 *Zn2*lgRn - 240.* Rn *Zn4*lgRn)* c_[6] + 
        2.* Rn *Zn *c_[8] + (-3. *Rn *Zn - 6.* Rn* Zn*lgRn)* c_[9] + (12. *Rn3* Zn - 8.* Rn *Zn3)* c_[10] + (-120. *Rn3* Zn - 80.* Rn *Zn3 + 240. *Rn3* Zn*lgRn - 
            160.* Rn *Zn3*lgRn) *c_[11]
          );
    }
    double R_0_, A_;
    std::vector<double> c_;
    Psip psip_;
};
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}\f]
 */ 
struct PsipRR
{
    /**
    * @brief Constructor
    *
    * @param gp geometric parameters
    */
    PsipRR( GeomParameters gp ): R_0_(gp.R_0), A_(gp.A), c_(gp.c), psip_(gp), psipR_(gp) {}
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}=
     \hat{R}_0^{-1} \Bigg\{ 2 c_2 +(3 \hat{\bar{R}}^2 )/2+2 c_9  \bar{Z}+c_4 (12 \bar{R}^2 -8  \bar{Z}^2)+c_{11} 
      (36 \bar{R}^2  \bar{Z}-8  \bar{Z}^3)
      +c_6 (30 \bar{R}^4 -144 \bar{R}^2  \bar{Z}^2+16  \bar{Z}^4)+c_3 (-3 -2  \ln{(\bar{R} 
      )})+
      A ((3 )/2-(3 \bar{R}^2 )/2+ \ln{(\bar{R}   )}) 
      +c_{10} (-9  \bar{Z}-6  
      \bar{Z} \ln{(\bar{R}   )})+c_5 (21 \bar{R}^2 -54  
      \bar{Z}^2+36 \bar{R}^2  \ln{(\bar{R}   )}-24  \bar{Z}^2 \ln{(\bar{R}   )})
      +c_{12} (-120 \bar{R}^2  \bar{Z}-240  \bar{Z}^3+720 \bar{R}^2  \bar{Z} \ln{(\bar{R}   )}
      -160  \bar{Z}^3 \ln{(\bar{R}   )})
      + c_7 (-165 \bar{R}^4 +2160 \bar{R}^2  \bar{Z}^2-640  \bar{Z}^4-450 \bar{R}^4  \ln{(\bar{R}   )}+2160 \bar{R}^2  \bar{Z}^2
      \ln{(\bar{R}   )}-240  \bar{Z}^4 \ln{(\bar{R}   )})\Bigg\}\f]
      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
      @return value
 */ 
    double operator()(double R, double Z) const
    {    
        //double psipR = psipR_.psipR_alt( R,Z);
        //return psip_.diff_psi_neu( R, Z)* psipRR_alt( R,Z) + psip_.diffdiff_psi_neu( R, Z) *psipR;
        return psipRR_alt( R, Z);
    }
    /**
    * @brief return operator()(R,Z)
    *
      @param R radius (cylindrical coordinates)
      @param Z height (cylindrical coordinates)
      @param phi angle (cylindrical coordinates)
    *
    * @return value
    */
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
    /**
    * @brief Display the internal parameters to std::cout
    */
    void display()
    {
      std::cout << R_0_ <<"  " <<A_ <<"\n";
      std::cout << c_[0] <<"\n";
    }
  private:
    double psipRR_alt(double R, double Z) const
    {    
       double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
       Rn = R/R_0_; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2;
       Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; 
       lgRn= log(Rn);
       return   1./R_0_*( (3.* Rn2)/2.*c_[12] + (3./2. - (3. *Rn2)/2. +lgRn) *A_ +  2.* c_[1] + (-3. - 2.*lgRn)* c_[2] + (12. *Rn2 - 8. *Zn2) *c_[3] + 
         (21. *Rn2 - 54. *Zn2 + 36. *Rn2*lgRn - 24. *Zn2*lgRn)* c_[4]
         + (30. *Rn4 - 144. *Rn2 *Zn2 + 16.*Zn4)*c_[5] + (-165. *Rn4 + 2160. *Rn2 *Zn2 - 640. *Zn4 - 450. *Rn4*lgRn + 
      2160. *Rn2 *Zn2*lgRn - 240. *Zn4*lgRn)* c_[6] + 
      2.* Zn* c_[8] + (-9. *Zn - 6.* Zn*lgRn) *c_[9] 
 +   (36. *Rn2* Zn - 8. *Zn3) *c_[10]
 +   (-120. *Rn2* Zn - 240. *Zn3 + 720. *Rn2* Zn*lgRn - 160. *Zn3*lgRn)* c_[11]);
    }
    double R_0_, A_;
    std::vector<double> c_;
    Psip psip_;
    PsipR psipR_;
};
/**
 * @brief \f[\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\f]
 */ 
struct PsipZ
{
    PsipZ( GeomParameters gp ): R_0_(gp.R_0), A_(gp.A), c_(gp.c), psip_(gp) { }
/**
 * @brief \f[\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}= 
      \Bigg\{c_8 +c_9 \bar{R}^2 +2 c_3  \bar{Z}-8 c_4 \bar{R}^2  \bar{Z}+c_{11} 
      (3 \bar{R}^4 -12 \bar{R}^2  \bar{Z}^2)+c_6 (-24 \bar{R}^4  \bar{Z}+32 \bar{R}^2  \bar{Z}^3)
      +c_{10} (3  \bar{Z}^2-3 \bar{R}^2 
      \ln{(\bar{R}   )})+c_5 (-18 \bar{R}^2  \bar{Z}+8  \bar{Z}^3-24 \bar{R}^2  \bar{Z}
      \ln{(\bar{R}   )})
      +c_{12} (-45 \bar{R}^4 +40  \bar{Z}^4+
      60 \bar{R}^4  \ln{(\bar{R}   )}-240 \bar{R}^2  \bar{Z}^2 \ln{(\bar{R}   )})
      +c_7 (150 \bar{R}^4  \bar{Z}-560 \bar{R}^2  \bar{Z}^3+48  
      \bar{Z}^5+360 \bar{R}^4  \bar{Z} \ln{(\bar{R}   )}-480 \bar{R}^2  \bar{Z}^3 \ln{(\bar{R}   )})\Bigg\} \f]
 */ 
    double psipZ_alt(double R, double Z) const
    {
        double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,Zn5,lgRn;
        Rn = R/R_0_; Rn2 = Rn*Rn;  Rn4 = Rn2*Rn2; 
        Zn = Z/R_0_; Zn2 = Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; Zn5 = Zn3*Zn2; 
        lgRn= log(Rn);

        return   (2.* Zn* c_[2] 
            -  8. *Rn2* Zn* c_[3] +
              ((-18.)*Rn2 *Zn + 8. *Zn3 - 24. *Rn2* Zn*lgRn) *c_[4] 
            + ((-24.) *Rn4* Zn + 32. *Rn2 *Zn3)* c_[5]   
            + (150. *Rn4* Zn - 560. *Rn2 *Zn3 + 48. *Zn5 + 360. *Rn4* Zn*lgRn - 480. *Rn2 *Zn3*lgRn)* c_[6] 
            + c_[7]
            + Rn2 * c_[8]
            + (3. *Zn2 - 3. *Rn2*lgRn)* c_[9]
            + (3. *Rn4 - 12. *Rn2 *Zn2) *c_[10]
            + ((-45.)*Rn4 + 40. *Zn4 + 60. *Rn4*lgRn -  240. *Rn2 *Zn2*lgRn)* c_[11]);
          
    }
    double operator()(double R, double Z) const
    {    
        //return psip_.diff_psi_neu( R, Z)* psipZ_alt( R,Z);
        return psipZ_alt(R, Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
    void display() const
    {
        std::cout << R_0_ <<"  " <<A_ <<"\n";
        std::cout << c_[0] <<"\n";
    }
  private:
    double R_0_, A_;
    std::vector<double> c_;
    Psip psip_;
};
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2}\f]
 */ 
struct PsipZZ
{
  PsipZZ( GeomParameters gp): R_0_(gp.R_0), A_(gp.A), c_(gp.c), psip_(gp), psipZ_(gp) { }
/**
 * @brief \f[ \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2}=
      \hat{R}_0^{-1} \Bigg\{2 c_3 -8 c_4 \bar{R}^2 +6 c_{10}  \bar{Z}-24 c_{11}
      \bar{R}^2  \bar{Z}+c_6 (-24 \bar{R}^4 +96 \bar{R}^2  \bar{Z}^2)
      +c_5 (-18 \bar{R}^2 +24  \bar{Z}^2-24 \bar{R}^2  \ln{(\bar{R}   )})+
      c_{12} (160  \bar{Z}^3-480 \bar{R}^2  \bar{Z} \ln{(\bar{R}   )})
      +c_7 (150 \bar{R}^4 -1680 \bar{R}^2  \bar{Z}^2+240  \bar{Z}^4+360 \bar{R}^4 
      \ln{(\bar{R}   )}-1440 \bar{R}^2  \bar{Z}^2 \ln{(\bar{R}   )})\Bigg\} \f]
 */ 
    double operator()(double R, double Z) const
    {    
        //double psipZ = psipZ_.psipZ_alt(R,Z);
        //return psip_.diff_psi_neu( R, Z)* psipZZ_alt( R,Z) + 
        //       psip_.diffdiff_psi_neu( R, Z) *psipZ*psipZ;
        return psipZZ_alt( R, Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
    void display() const
    {
        std::cout << R_0_ <<"  " <<A_ <<"\n";
        std::cout << c_[0] <<"\n";
    }
  private:
    double psipZZ_alt(double R, double Z) const
    {
        double Rn,Rn2,Rn4,Zn,Zn2,Zn3,Zn4,lgRn;
        Rn = R/R_0_; Rn2 = Rn*Rn; Rn4 = Rn2*Rn2; 
        Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; Zn4 = Zn2*Zn2; 
        lgRn= log(Rn);
        return   1./R_0_*( 2.* c_[2] - 8. *Rn2* c_[3] + (-18. *Rn2 + 24. *Zn2 - 24. *Rn2*lgRn) *c_[4] + (-24.*Rn4 + 96. *Rn2 *Zn2) *c_[5]
        + (150. *Rn4 - 1680. *Rn2 *Zn2 + 240. *Zn4 + 360. *Rn4*lgRn - 1440. *Rn2 *Zn2*lgRn)* c_[6] + 6.* Zn* c_[9] -  24. *Rn2 *Zn *c_[10] + (160. *Zn3 - 480. *Rn2* Zn*lgRn) *c_[11]);
    }
    double R_0_, A_;
    std::vector<double> c_;
    Psip psip_; 
    PsipZ psipZ_;
};
/**
 * @brief  \f[\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\f] 
 */ 
struct PsipRZ
{
    PsipRZ( GeomParameters gp ): R_0_(gp.R_0), A_(gp.A), c_(gp.c), psip_(gp), psipZ_(gp), psipR_(gp) {  }
/**
 * @brief  \f[\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}= 
        \hat{R}_0^{-1} \Bigg\{2 c_9 \bar{R} -16 c_4 \bar{R}  \bar{Z}+c_{11} 
      (12 \bar{R}^3 -24 \bar{R}  \bar{Z}^2)+c_6 (-96 \bar{R}^3  \bar{Z}+64 \bar{R}  \bar{Z}^3)
      + c_{10} (-3 \bar{R} -6 \bar{R}  \ln{(\bar{R}   )})
      +c_5 (-60 \bar{R}  \bar{Z}-48 \bar{R}  \bar{Z} \ln{(\bar{R}   )})
      +c_{12}  (-120 \bar{R}^3 -240 \bar{R}  \bar{Z}^2+
      240 \bar{R}^3  \ln{(\bar{R}   )}-480 \bar{R}  \bar{Z}^2 \ln{(\bar{R}   )})
      +c_7(960 \bar{R}^3  \bar{Z}-1600 \bar{R}  \bar{Z}^3+1440 \bar{R}^3  \bar{Z} \ln{(\bar{R} 
      )}-960 \bar{R}  \bar{Z}^3 \ln{(\bar{R}   )})\Bigg\} \f] 
 */ 
    double operator()(double R, double Z) const
    {    
        //return psip_.diff_psi_neu( R, Z)* psipRZ_alt( R,Z) + 
        //       psip_.diffdiff_psi_neu( R, Z) *psipR_.psipR_alt(R,Z)*psipZ_.psipZ_alt(R,Z);
        return psipRZ_alt( R, Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
    void display() const
    {
        std::cout << R_0_ <<"  " <<A_ <<"\n";
        std::cout << c_[0] <<"\n";
    }
  private:
    double psipRZ_alt(double R, double Z) const
    {
        double Rn,Rn2,Rn3,Zn,Zn2,Zn3,lgRn;
        Rn = R/R_0_; Rn2 = Rn*Rn; Rn3 = Rn2*Rn; 
        Zn = Z/R_0_; Zn2 =Zn*Zn; Zn3 = Zn2*Zn; 
        lgRn= log(Rn);
        return   1./R_0_*(
              -16.* Rn* Zn* c_[3] + (-60.* Rn* Zn - 48.* Rn* Zn*lgRn)* c_[4] + (-96. *Rn3* Zn + 64.*Rn *Zn3)* c_[5]
            + (960. *Rn3 *Zn - 1600.* Rn *Zn3 + 1440. *Rn3* Zn*lgRn - 960. *Rn *Zn3*lgRn) *c_[6] +  2.* Rn* c_[8] + (-3.* Rn - 6.* Rn*lgRn)* c_[9]
            + (12. *Rn3 - 24.* Rn *Zn2) *c_[10] + (-120. *Rn3 - 240. *Rn *Zn2 + 240. *Rn3*lgRn -   480.* Rn *Zn2*lgRn)* c_[11]
                 );
    }
    double R_0_, A_;
    std::vector<double> c_;
    Psip psip_; 
    PsipZ psipZ_;
    PsipR psipR_;
};
/**
 * @brief \f[\hat{I}\f] 
 */ 
struct Ipol
{
    Ipol(  GeomParameters gp ):  R_0_(gp.R_0), A_(gp.A), qampl_(gp.qampl), psip_(gp) { }
    /**
    * @brief \f[\hat{I}= \sqrt{-2 A \hat{\psi}_p / \hat{R}_0 +1}\f] 
    */ 
    double operator()(double R, double Z) const
    {    
        //sign before A changed to -
        return qampl_*sqrt(-2.*A_* psip_(R,Z) /R_0_ + 1.);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        //sign before A changed to -
      return  qampl_*sqrt(-2.*A_*psip_(R,Z,phi)/R_0_ + 1.);
    }
    void display() const
    {
      std::cout<< R_0_ <<"  "  << A_ <<"\n";
    }
    private:
    double R_0_, A_,qampl_;
    Psip psip_;
};

/**
 * @brief \f[   \frac{1}{\hat{B}}   \f]
 */ 
struct InvB
{
    InvB(  GeomParameters gp ):  R_0_(gp.R_0), ipol_(gp), psipR_(gp), psipZ_(gp)  { }
    /**
    * @brief \f[   \frac{1}{\hat{B}} = 
        \frac{\hat{R}}{\hat{R}_0}\frac{1}{ \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
        + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2}}  \f]
    */ 
    double operator()(double R, double Z) const
    {    
        double psipR = psipR_(R,Z), psipZ = psipZ_(R,Z), ipol = ipol_(R,Z);
        return R/(R_0_*sqrt(ipol*ipol + psipR*psipR +psipZ*psipZ)) ;
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
        return operator()(R,Z);
    }
  private:
    double R_0_;
    Ipol ipol_;
    PsipR psipR_;
    PsipZ psipZ_;  
};
/**
 * @brief \f[   \ln{(   \hat{B})}  \f]
 */ 
struct LnB
{
    LnB( GeomParameters gp ):  R_0_(gp.R_0), ipol_(gp), psipR_(gp), psipZ_(gp)  { }
/**
 * @brief \f[   \ln{(   \hat{B})} = \ln{\left[
      \frac{\hat{R}_0}{\hat{R}} \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
      + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2} \right] } \f]
 */ 
    double operator()(double R, double Z) const
    {    
      return log((R_0_*sqrt(ipol_(R,Z)*ipol_(R,Z) + psipR_(R,Z)*psipR_(R,Z) +psipZ_(R,Z)*psipZ_(R,Z)))/R) ;
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()(double R, double Z, double phi) const
    {    
      return log((R_0_*sqrt(ipol_(R,Z,phi)*ipol_(R,Z,phi) + psipR_(R,Z,phi)*psipR_(R,Z,phi) +psipZ_(R,Z,phi)*psipZ_(R,Z,phi)))/R) ;
    }
  private:
    double R_0_;
    Ipol ipol_;
    PsipR psipR_;
    PsipZ psipZ_;  
};
/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{R}}  \f]
 */ 
struct BR
{
    BR(GeomParameters gp):  R_0_(gp.R_0), A_(gp.A), qampl_(gp.qampl), psipR_(gp), psipRR_(gp),psipZ_(gp) ,psipRZ_(gp), invB_(gp) { }
/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{R}} = 
      -\frac{\hat{R}^2\hat{R}_0^{-2} \hat{B}^2+A\hat{R} \hat{R}_0^{-1}   \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)  
      - \hat{R} \left[  \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\right)
      + \left( \frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)\left( \frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R}^2}\right)\right] }{\hat{R}^3 \hat{R}_0^{-2}\hat{B}} \f]
 */ 
  double operator()(double R, double Z) const
  { 
    double Rn;
    Rn = R/R_0_;
    //sign before A changed to +
    return -( Rn*Rn/invB_(R,Z)/invB_(R,Z)+ qampl_*qampl_*Rn *A_*psipR_(R,Z) - R  *(psipZ_(R,Z)*psipRZ_(R,Z)+psipR_(R,Z)*psipRR_(R,Z)))/(R*Rn*Rn/invB_(R,Z));
  }
    /**
     * @brief == operator()(R,Z)
     */ 
  double operator()(double R, double Z, double phi) const
  { 
    double Rn;
    Rn = R/R_0_;
    //sign before A changed to +
    return -( Rn*Rn/invB_(R,Z,phi)/invB_(R,Z,phi)+ qampl_*qampl_*Rn *A_*psipR_(R,Z,phi) - R *(psipZ_(R,Z,phi)*psipRZ_(R,Z,phi)+psipR_(R,Z,phi)*psipRR_(R,Z,phi)))/(R*Rn*Rn/invB_(R,Z,phi));
  }
  private:
    double R_0_;
    double A_;
    double qampl_;
    PsipR psipR_;
    PsipRR psipRR_;
    PsipZ psipZ_;
    PsipRZ psipRZ_;  
    InvB invB_;
};
/**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{Z}}  \f]
 */ 
struct BZ
{
    BZ( GeomParameters gp):  R_0_(gp.R_0), A_(gp.A), qampl_(gp.qampl), psipR_(gp),psipZ_(gp), psipZZ_(gp) ,psipRZ_(gp), invB_(gp) { }
  /**
 * @brief \f[  \frac{\partial \hat{B} }{ \partial \hat{Z}} = 
 \frac{-A \hat{R}_0^{-1}    \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}   \right)+
 \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{R} \partial\hat{Z}}\right)
      + \left( \frac{\partial \hat{\psi}_p }{ \partial \hat{Z}} \right)\left(\frac{\partial^2  \hat{\psi}_p }{ \partial \hat{Z}^2} \right)}{\hat{R}^2 \hat{R}_0^{-2}\hat{B}} \f]
 */ 
  double operator()(double R, double Z) const
  { 
    double Rn;
    Rn = R/R_0_;
    //sign before A changed to -
    return (-qampl_*qampl_*A_/R_0_*psipZ_(R,Z) + psipR_(R,Z)*psipRZ_(R,Z)+psipZ_(R,Z)*psipZZ_(R,Z))/(Rn*Rn/invB_(R,Z));
  }
    /**
     * @brief == operator()(R,Z)
     */ 
  double operator()(double R, double Z, double phi) const
  { 
      //sign before A changed to -
    double Rn;
    Rn = R/R_0_;
    return (-qampl_*qampl_*A_/R_0_*psipZ_(R,Z,phi) + psipR_(R,Z,phi)*psipRZ_(R,Z,phi)+psipZ_(R,Z,phi)*psipZZ_(R,Z,phi))/(Rn*Rn/invB_(R,Z,phi));
  }
  private:
    double R_0_;
    double A_;
    double qampl_;
    PsipR psipR_;
    PsipZ psipZ_;
    PsipZZ psipZZ_;
    PsipRZ psipRZ_;  
    InvB invB_; 
};


/**
 * @brief \f[ \mathcal{\hat{K}}^{\hat{R}}  \f]
 */ 
struct CurvatureR
{
    CurvatureR( GeomParameters gp):
        gp_(gp),
        psip_(gp),
        psipR_(gp), psipZ_(gp),
        psipZZ_(gp), psipRZ_(gp),
        ipol_(gp),
        invB_(gp),
        bZ_(gp) { }
/**
 * @brief \f[ \mathcal{\hat{K}}^{\hat{R}} =-\frac{1}{ \hat{B}^2}  \frac{\partial \hat{B}}{\partial \hat{Z}}  \f]
 */ 
    double operator()( double R, double Z) const
    {
        return -2.*invB_(R,Z)*invB_(R,Z)*bZ_(R,Z); //factor 2 stays under discussion
//         return -ipol_(R,Z)*invB_(R,Z)*invB_(R,Z)*invB_(R,Z)*bZ_(R,Z)*gp_.R_0/R; //factor 2 stays under discussion

    }
    
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return -2.*invB_(R,Z,phi)*invB_(R,Z,phi)*bZ_(R,Z,phi); //factor 2 stays under discussion
//         return -ipol_(R,Z,phi)*invB_(R,Z,phi)*invB_(R,Z,phi)*invB_(R,Z,phi)*bZ_(R,Z,phi)*gp_.R_0/R; //factor 2 stays under discussion
    }
    private:    
    GeomParameters gp_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipZ  psipZ_;
    PsipZZ psipZZ_;
    PsipRZ psipRZ_;
    Ipol   ipol_;
    InvB   invB_;
    BZ bZ_;    
};
/**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}}  \f]
 */ 
struct CurvatureZ
{
    CurvatureZ( GeomParameters gp):
        gp_(gp),
        psip_(gp),
        psipR_(gp),
        psipRR_(gp),
        psipZ_(gp),
        psipRZ_(gp),
        ipol_(gp),
        invB_(gp),
        bR_(gp) { }
 /**
 * @brief \f[  \mathcal{\hat{K}}^{\hat{Z}} =\frac{1}{ \hat{B}^2}   \frac{\partial \hat{B}}{\partial \hat{R}} \f]
 */    
    double operator()( double R, double Z) const
    {
        return 2.*invB_(R,Z)*invB_(R,Z)*bR_(R,Z); //factor 2 stays under discussion
//         return  ipol_(R,Z)*invB_(R,Z)*invB_(R,Z)*invB_(R,Z)*bR_(R,Z)*gp_.R_0/R; //factor 2 stays under discussion
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return 2.*invB_(R,Z,phi)*invB_(R,Z,phi)*bR_(R,Z,phi); //factor 2 stays under discussion
//         return ipol_(R,Z,phi)*invB_(R,Z,phi)*invB_(R,Z,phi)*invB_(R,Z,phi)*bR_(R,Z,phi)*gp_.R_0/R; //factor 2 stays under discussion
    }
    private:    
    GeomParameters gp_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipRR  psipRR_;
    PsipZ  psipZ_;    
    PsipRZ psipRZ_;
    Ipol   ipol_;
    InvB   invB_;
    BR bR_;   
};

/**
 * @brief \f[  \hat{\nabla}_\parallel \ln{(\hat{B})} \f]
 */ 
struct GradLnB
{
    GradLnB( GeomParameters gp):
        gp_(gp),
        psip_(gp),
        psipR_(gp),
        psipRR_(gp),
        psipZ_(gp),
        psipZZ_(gp),
        psipRZ_(gp),
        ipol_(gp),
        invB_(gp),
        bR_(gp), 
        bZ_(gp) {

    } 
    /**
 * @brief \f[  \hat{\nabla}_\parallel \ln{(\hat{B})} = \frac{1}{\hat{R}\hat{B}^2 } \left[ \hat{B}, \hat{\psi}_p\right]_{\hat{R}\hat{Z}} \f]
 */ 
    double operator()( double R, double Z) const
    {
       return gp_.R_0*invB_(R,Z)*invB_(R,Z)*(bR_(R,Z) *psipZ_(R,Z) - bZ_(R,Z)* psipR_(R,Z))/R ;
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
       return gp_.R_0*invB_(R,Z,phi)* invB_(R,Z,phi)*(bR_(R,Z,phi) *psipZ_(R,Z,phi) - bZ_(R,Z,phi)* psipR_(R,Z,phi))/R ;
    }
    private:
    GeomParameters gp_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipRR  psipRR_;
    PsipZ  psipZ_;    
    PsipZZ  psipZZ_;    
    PsipRZ psipRZ_;
    Ipol   ipol_;
    InvB   invB_;
    BR bR_;
    BZ bZ_;   
};
/**
 * @brief Integrates the equations for a field line and 1/B
 */ 
struct Field
{
    Field( GeomParameters gp):
        gp_(gp),
        psipR_(gp),
        psipZ_(gp),
        ipol_(gp),
        invB_(gp) {
    }
    /**
 * @brief \f[ \frac{d \hat{R} }{ d \varphi}  = \frac{\hat{R}}{\hat{I}} \frac{\partial\hat{\psi}_p}{\partial \hat{Z}}, \hspace {3 mm}
 \frac{d \hat{Z} }{ d \varphi}  =- \frac{\hat{R}}{\hat{I}} \frac{\partial \hat{\psi}_p}{\partial \hat{R}} , \hspace {3 mm}
 \frac{d \hat{l} }{ d \varphi}  =\frac{\hat{R}^2 \hat{B}}{\hat{I} \hat{R}_0}  \f]
 */ 
    void operator()( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp) const
    {
        for( unsigned i=0; i<y[0].size(); i++)
        {
            yp[2][i] =  y[0][i]*y[0][i]/invB_(y[0][i],y[1][i])/ipol_(y[0][i],y[1][i])/gp_.R_0;//ds/dphi =  R^2 B/I/R_0_hat
            yp[0][i] =  y[0][i]*psipZ_(y[0][i],y[1][i])/ipol_(y[0][i],y[1][i]);               //dR/dphi =  R/I Psip_Z
            yp[1][i] = -y[0][i]*psipR_(y[0][i],y[1][i])/ipol_(y[0][i],y[1][i]) ;              //dZ/dphi = -R/I Psip_R
        }
    }
    /**
 * @brief \f[ \frac{d \hat{R} }{ d \varphi}  = \frac{\hat{R}}{\hat{I}} \frac{\partial\hat{\psi}_p}{\partial \hat{Z}}, \hspace {3 mm}
 \frac{d \hat{Z} }{ d \varphi}  =- \frac{\hat{R}}{\hat{I}} \frac{\partial \hat{\psi}_p}{\partial \hat{R}} , \hspace {3 mm}
 \frac{d \hat{l} }{ d \varphi}  =\frac{\hat{R}^2 \hat{B}}{\hat{I}  \hat{R}_0}  \f]
 */ 
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double ipol = ipol_(y[0],y[1]);
        yp[2] =  y[0]*y[0]/invB_(y[0],y[1])/ipol/gp_.R_0;       //ds/dphi =  R^2 B/I/R_0_hat
        yp[0] =  y[0]*psipZ_(y[0],y[1])/ipol;              //dR/dphi =  R/I Psip_Z
        yp[1] = -y[0]*psipR_(y[0],y[1])/ipol ;             //dZ/dphi = -R/I Psip_R

    }
    /**
     * @brief \f[   \frac{1}{\hat{B}} = 
      \frac{\hat{R}}{\hat{R}_0}\frac{1}{ \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
      + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2}}  \f]
     */ 
    double operator()( double R, double Z) const
    {
        //modified
//          return invB_(R,Z)* invB_(R,Z)*ipol_(R,Z)*gp_.R_0/R;
        return invB_(R,Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return invB_(R,Z,phi);

//         return invB_(R,Z,phi)*invB_(R,Z,phi)*ipol_(R,Z,phi)*gp_.R_0/R;
    }
    
    private:
    GeomParameters gp_;
    PsipR  psipR_;
    PsipZ  psipZ_;
    Ipol   ipol_;
    InvB   invB_;
   
};

/**
 * @brief Phi component of magnetic field \f$ B_\Phi\f$
 */
struct FieldP
{
    FieldP( GeomParameters gp): R_0(gp.R_0), 
    ipol_(gp){}
    double operator()( double R, double Z, double phi) const
    {
        return R_0*ipol_(R,Z)/R/R;
    }
    
    private:
    double R_0;
    Ipol   ipol_;
   
}; 
/**
 * @brief R component of magnetic field\f$ B_R\f$
 */
struct FieldR
{
    FieldR( GeomParameters gp): psipZ_(gp), R_0(gp.R_0){ }
    double operator()( double R, double Z) const
    {
        return  R_0/R*psipZ_(R,Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return  this->operator()(R,Z);
    }
    private:
    PsipZ  psipZ_;
    double R_0;
   
};
/**
 * @brief Z component of magnetic field \f$ B_Z\f$
 */
struct FieldZ
{
    FieldZ( GeomParameters gp): psipR_(gp), R_0(gp.R_0){ }
    double operator()( double R, double Z) const
    {
        return  -R_0/R*psipR_(R,Z);
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return  this->operator()(R,Z);
    }
    private:
    PsipR  psipR_;
    double R_0;
   
};

/**
 * @brief \f$\theta\f$ component of magnetic field 
 */ 
struct FieldT
{
    FieldT( GeomParameters gp):  R_0_(gp.R_0), fieldR_(gp), fieldZ_(gp){}
  /**
 * @brief \f[  B^{\theta} = 
 * B^R\partial_R\theta + B^Z\partial_Z\theta\f]
 * where \f$ \theta \f$ is the geometrical poloidal angle.
 */ 
  double operator()(double R, double Z) const
  { 
      double r2 = (R-R_0_)*(R-R_0_) + Z*Z;
      return fieldR_(R,Z)*(-Z/r2) + fieldZ_(R,Z)*(R-R_0_)/r2;
  }
    /**
     * @brief == operator()(R,Z)
     */ 
  double operator()(double R, double Z, double phi) const
  { 
      return this->operator()(R,Z);
  }
  private:
    double R_0_;
    FieldR fieldR_;
    FieldZ fieldZ_;

};

/**
 * @brief y-component of magnetic field 
 */ 
struct FieldY
{
    FieldY( GeomParameters gp): f_psi_(1.), R_0_(gp.R_0), psipR_(gp), psipZ_(gp){}
    void set_f( double f_new){f_psi_=f_new;}
  /**
 * @brief \f[  B^y= 
 * f(\psi) \frac{(\nabla\psi)^2}{R} \f]
 */ 
    double operator()(double R, double Z) const
    { 
        double psipR = psipR_(R, Z), psipZ = psipZ_(R,Z);
        double psi2 = psipR*psipR+ psipZ*psipZ;
        return f_psi_*(psi2)*R_0_/R;
    }
      /**
       * @brief == operator()(R,Z)
       */ 
    double operator()(double R, double Z, double phi) const
    { 
        return this->operator()(R,Z);
    }
    private:
    double f_psi_, R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
};

struct FieldRZYT
{
    FieldRZYT( GeomParameters gp): R_0_(gp.R_0), psipR_(gp), psipZ_(gp){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        yp[0] =  R_0_/y[0]*psipZ;//fieldR
        yp[1] = -R_0_/y[0]*psipR;//fieldZ
        yp[2] = (psipR*psipR+psipZ*psipZ)*R_0_/y[0]; //fieldYbar
        double r2 = (y[0]-R_0_)*(y[0]-R_0_) + y[1]*y[1];
        double fieldT = yp[0]*(-y[1]/r2) + yp[1]*(y[0]-R_0_)/r2; //fieldT
        //if( y[0] <= 0){ std::cerr << "Attention R is zero!!\n";
        //    std::cerr << y[0]<<" "<<y[1]<<"\n";
        //    std::cerr << psipR<<" "<<psipZ<<"\n";

        //}
        //if( fieldT == 0) std::cerr << "Attention fieldT is zero!!\n";
        yp[0] /=  fieldT;
        yp[1] /=  fieldT;
        yp[2] /=  fieldT;
    }
  private:
    double R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
};

struct FieldRZY
{
    FieldRZY( GeomParameters gp): R_0_(gp.R_0), psipR_(gp), psipZ_(gp){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        yp[0] =  R_0_/y[0]*psipZ;//fieldR
        yp[1] = -R_0_/y[0]*psipR;//fieldZ
        yp[2] = (psipR*psipR+psipZ*psipZ)*R_0_/y[0]; //fieldYbar
        //double r2 = (y[0]-R_0_)*(y[0]-R_0_) + y[1]*y[1];
        //double fieldT = yp[0]*(-y[1]/r2) + yp[1]*(y[0]-R_0_)/r2; //fieldT
        yp[0] /=  yp[1];
        yp[2] /=  yp[1];
        yp[1] =  1.;
    }
  private:
    double R_0_;
    PsipR psipR_;
    PsipZ psipZ_;
};

struct FieldRZtau
{
    FieldRZtau( GeomParameters gp): psipR_(gp), psipZ_(gp){}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psi2 = psipR*psipR+ psipZ*psipZ;
        yp[0] =  psipR/psi2;
        yp[1] =  psipZ/psi2;
    }
  private:
    PsipR psipR_;
    PsipZ psipZ_;
};

struct HessianRZtau
{
    HessianRZtau( GeomParameters gp): norm_(false), quad_(1), psipR_(gp), psipZ_(gp), psipRR_(gp), psipRZ_(gp), psipZZ_(gp){}
    // if true goes into positive Z - direction and X else
    void set_quadrant( int quadrant) {quad_ = quadrant;}
    void set_norm( bool normed) {norm_ = normed;}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipRZ = psipRZ_(y[0], y[1]);
        if( psipRZ == 0)
        {
            if(      quad_ == 0) { yp[0] = 1; yp[1] = 0; }
            else if( quad_ == 1) { yp[0] = 0; yp[1] = 1; }
            else if( quad_ == 2) { yp[0] = -1; yp[1] = 0; }
            else if( quad_ == 3) { yp[0] = 0; yp[1] = -1; }
        }
        else
        {
            double psipRR = psipRR_(y[0], y[1]), psipZZ = psipZZ_(y[0],y[1]);
            double T = psipRR + psipZZ; 
            double D = psipZZ*psipRR - psipRZ*psipRZ;
            double L1 = 0.5*T+sqrt( 0.25*T*T-D); // > 0
            double L2 = 0.5*T-sqrt( 0.25*T*T-D); // < 0;  D = L1*L2
            if      ( quad_ == 0){ yp[0] =  L1 - psipZZ; yp[1] = psipRZ;}
            else if ( quad_ == 1){ yp[0] = -psipRZ; yp[1] = psipRR - L2;}
            else if ( quad_ == 2){ yp[0] =  psipZZ - L1; yp[1] = -psipRZ;}
            else if ( quad_ == 3){ yp[0] = +psipRZ; yp[1] = L2 - psipRR;}
        }
        if( norm_) 
        {
            double vgradpsi = yp[0]*psipR_(y[0],y[1]) + yp[1]*psipZ_(y[0],y[1]);
            yp[0] /= vgradpsi, yp[1] /= vgradpsi;
        }
        else
        {
            double norm = sqrt(yp[0]*yp[0]+yp[1]*yp[1]);
            yp[0]/= norm, yp[1]/= norm;
        }

    }
  private:
    bool norm_;
    int quad_;
    PsipR psipR_;
    PsipZ psipZ_;
    PsipRR psipRR_;
    PsipRZ psipRZ_;
    PsipZZ psipZZ_;
};

struct FieldRZYRYZY
{
    FieldRZYRYZY( const GeomParameters& gp): psipR_(gp), psipZ_(gp), psipRR_(gp), psipRZ_(gp), psipZZ_(gp){ f_ = f_prime_ = 1.;}
    void set_f( double new_f){ f_ = new_f;}
    void set_fp( double new_fp){ f_prime_ = new_fp;}
    void operator()( const dg::HVec& y, dg::HVec& yp) const
    {
        double psipR = psipR_(y[0], y[1]), psipZ = psipZ_(y[0],y[1]);
        double psipRR = psipRR_(y[0], y[1]), psipRZ = psipRZ_(y[0],y[1]), psipZZ = psipZZ_(y[0],y[1]);
        double psip2 = psipR*psipR+ psipZ*psipZ;

        yp[0] =  psipZ/f_/psip2;
        yp[1] =  -psipR/f_/psip2;
        yp[2] =  ( (psipZ/y[0] - psipRZ )*y[2] 
                   +( psipRR - psipR/y[0])*y[3] )/f_/psip2 
            + f_prime_/f_* psipR + 2.*(psipR*psipRR + psipZ*psipRZ)/psip2 -1./y[0];
        yp[3] =  (psipRZ*y[3] - psipZZ*y[2])/f_/psip2 
            + f_prime_/f_* psipZ + 2.*(psipR*psipRZ + psipZ*psipZZ)/psip2;
    }
  private:
    double f_, f_prime_;
    PsipR psipR_;
    PsipZ psipZ_;
    PsipRR psipRR_;
    PsipRZ psipRZ_;
    PsipZZ psipZZ_;
};

/**
 * @brief R component of magnetic field\f$ b_R\f$
 */
struct BHatR
{
    BHatR( GeomParameters gp): 
        psipZ_(gp), R_0(gp.R_0),
        invB_(gp){ }
    double operator()( double R, double Z, double phi) const
    {
        return  invB_(R,Z)*R_0/R*psipZ_(R,Z);
    }
    private:
    PsipZ  psipZ_;
    double R_0;
    InvB   invB_;

};
/**
 * @brief Z component of magnetic field \f$ b_Z\f$
 */
struct BHatZ
{
    BHatZ( GeomParameters gp): 
        psipR_(gp), R_0(gp.R_0),
        invB_(gp){ }

    double operator()( double R, double Z, double phi) const
    {
        return  -invB_(R,Z)*R_0/R*psipR_(R,Z);
    }
    private:
    PsipR  psipR_;
    double R_0;
    InvB   invB_;

};
/**
 * @brief Phi component of magnetic field \f$ b_\Phi\f$
 */
struct BHatP
{
    BHatP( GeomParameters gp):
        R_0(gp.R_0), 
        ipol_(gp),
        invB_(gp){}
    double operator()( double R, double Z, double phi) const
    {
        return invB_(R,Z)*R_0*ipol_(R,Z)/R/R;
    }
    
    private:
    double R_0;
    Ipol   ipol_;
    InvB   invB_;
  
}; 
///@} 
///@addtogroup profiles
///@{

/**
 * @brief Delta function for poloidal flux \f$ B_Z\f$
     \f[ |\nabla \psi_p|\delta(\psi_p(R,Z)-\psi_0) = \frac{\sqrt{ (\nabla \psi_p)^2}}{\sqrt{2\pi\varepsilon}} \exp\left(-\frac{(\psi_p(R,Z) - \psi_{0})^2}{2\varepsilon} \right)  \f]
 */
struct DeltaFunction
{
    DeltaFunction(GeomParameters gp, double epsilon,double psivalue) :
        psip_(gp),
        psipR_(gp), 
        psipZ_(gp), 
        epsilon_(epsilon),
        psivalue_(psivalue){
    }
    /**
    * @brief Set a new \f$ \varepsilon\f$
    *
    * @param eps new value
    */
    void setepsilon(double eps ){epsilon_ = eps;}
    /**
    * @brief Set a new \f$ \psi_0\f$
    *
    * @param psi_0 new value
    */
    void setpsi(double psi_0 ){psivalue_ = psi_0;}

    /**
     *@brief \f[ \frac{\sqrt{ (\nabla \psi_p)^2}}{\sqrt{2\pi\varepsilon}} \exp\left(-\frac{(\psi_p(R,Z) - \psi_{0})^2}{2\varepsilon} \right)  \f]
     */
    double operator()( double R, double Z) const
    {
        return 1./sqrt(2.*M_PI*epsilon_)*
               exp(-( (psip_(R,Z)-psivalue_)* (psip_(R,Z)-psivalue_))/2./epsilon_)*sqrt(psipR_(R,Z)*psipR_(R,Z) +psipZ_(R,Z)*psipZ_(R,Z));
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return (*this)(R,Z);
    }
    private:
    Psip psip_;
    PsipR  psipR_;
    PsipZ  psipZ_;
    double epsilon_;
    double psivalue_;
};

/**
 * @brief Flux surface average over quantity
 *
 * The average is computed using the formula
 \f[ <f>(\psi_0) = \frac{1}{A} \int dV \delta(\psi_p(R,Z)-\psi_0) |\nabla\psi_p|f(R,Z) \f]
 with \f$ A = \int dV \delta(\psi_p(R,Z)-\psi_0)|\nabla\psi_p|\f$
 * @tparam container  The container class of the vector to average

 */
template <class container = thrust::host_vector<double> >
struct FluxSurfaceAverage
{
     /**
     * @brief Construct from a field and a grid
     * @param g2d 2d grid
     * @param gp  geometry parameters
     * @param f container for global safety factor
     */
    FluxSurfaceAverage(const dg::Grid2d<double>& g2d, GeomParameters gp,   const container& f) :
    g2d_(g2d),
    gp_(gp),
    f_(f),
    psip_(gp),
    psipR_(gp),
    psipZ_(gp),
    deltaf_(DeltaFunction(gp,0.0,0.0)),
    w2d_ ( dg::create::weights( g2d_)),
    oneongrid_(dg::evaluate(dg::one,g2d_))              
    {
      dg::HVec psipRog2d  = dg::evaluate( psipR_, g2d_);
      dg::HVec psipZog2d  = dg::evaluate( psipZ_, g2d_);
      double psipRmax = (double)thrust::reduce( psipRog2d.begin(), psipRog2d.end(),  0.,     thrust::maximum<double>()  );    
      //double psipRmin = (double)thrust::reduce( psipRog2d.begin(), psipRog2d.end(),  psipRmax,thrust::minimum<double>()  );
      double psipZmax = (double)thrust::reduce( psipZog2d.begin(), psipZog2d.end(), 0.,      thrust::maximum<double>()  );    
      //double psipZmin = (double)thrust::reduce( psipZog2d.begin(), psipZog2d.end(), psipZmax,thrust::minimum<double>()  );   
      double deltapsi = fabs(psipZmax/g2d_.Ny()/g2d_.n() +psipRmax/g2d_.Nx()/g2d_.n());
      //deltaf_.setepsilon(deltapsi/4.);
      deltaf_.setepsilon(deltapsi); //macht weniger Zacken
    }
    /**
     * @brief Calculate the Flux Surface Average
     *
     * @param psip0 the actual psi value for q(psi)
     */
    double operator()(double psip0)
    {
        deltaf_.setpsi( psip0);
        container deltafog2d = dg::evaluate( deltaf_, g2d_);    
        double psipcut = dg::blas2::dot( f_,w2d_,deltafog2d); //int deltaf psip
        double vol     = dg::blas2::dot( oneongrid_ , w2d_,deltafog2d); //int deltaf
        double fsa = psipcut/vol;
        return fsa;
    }
    private:
    dg::Grid2d<double> g2d_;
    GeomParameters gp_;    
    container f_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipZ  psipZ_;
    DeltaFunction deltaf_;    
    const container w2d_;
    const container oneongrid_;
};
/**
 * @brief Class for the evaluation of the safety factor q
 * \f[ q(\psi_0) = \frac{1}{2\pi} \int dV |\nabla\psi_p| \delta(\psi_p-\psi_0) \alpha( R,Z) \f]
 * @tparam container 
 *
 */
template <class container = thrust::host_vector<double> >
struct SafetyFactor
{
     /**
     * @brief Construct from a field and a grid
     * @param g2d 2d grid
     * @param gp  geometry parameters
     * @param f container for global safety factor
     */
    SafetyFactor(const dg::Grid2d<double>& g2d, GeomParameters gp,   const container& f) :
    g2d_(g2d),
    gp_(gp),
    f_(f), //why not directly use Alpha??
    psip_(gp),
    psipR_(gp),
    psipZ_(gp),
    deltaf_(DeltaFunction(gp,0.0,0.0)),
    w2d_ ( dg::create::weights( g2d_)),
    oneongrid_(dg::evaluate(dg::one,g2d_))              
    {
      dg::HVec psipRog2d  = dg::evaluate( psipR_, g2d_);
      dg::HVec psipZog2d  = dg::evaluate( psipZ_, g2d_);
      double psipRmax = (double)thrust::reduce( psipRog2d.begin(), psipRog2d.end(), 0.,     thrust::maximum<double>()  );    
      //double psipRmin = (double)thrust::reduce( psipRog2d.begin(), psipRog2d.end(),  psipRmax,thrust::minimum<double>()  );
      double psipZmax = (double)thrust::reduce( psipZog2d.begin(), psipZog2d.end(), 0.,      thrust::maximum<double>()  );    
      //double psipZmin = (double)thrust::reduce( psipZog2d.begin(), psipZog2d.end(), psipZmax,thrust::minimum<double>()  );   
      double deltapsi = fabs(psipZmax/g2d_.Ny() +psipRmax/g2d_.Nx());
      //deltaf_.setepsilon(deltapsi/4.);
      deltaf_.setepsilon(4.*deltapsi); //macht weniger Zacken
    }
    /**
     * @brief Calculate the q profile over the function f which has to be the global safety factor
     * \f[ q(\psi_0) = \frac{1}{2\pi} \int dV |\nabla\psi_p| \delta(\psi_p-\psi_0) \alpha( R,Z) \f]
     *
     * @param psip0 the actual psi value for q(psi)
     */
    double operator()(double psip0)
    {
        deltaf_.setpsi( psip0);
        container deltafog2d = dg::evaluate( deltaf_, g2d_);    
        double q = dg::blas2::dot( f_,w2d_,deltafog2d)/(2.*M_PI);
        return q;
    }
    private:
    dg::Grid2d<double> g2d_;
    GeomParameters gp_;    
    container f_;
    Psip   psip_;    
    PsipR  psipR_;
    PsipZ  psipZ_;
    DeltaFunction deltaf_;    
    const container w2d_;
    const container oneongrid_;
};
/**
 * @brief Global safety factor
\f[ \alpha(R,Z) = \frac{|B^\varphi|}{R|B^\eta|} = \frac{I_{pol}(R,Z)}{R|\nabla\psi_p|} \f]
 */
struct Alpha
{
    Alpha( GeomParameters gp):
        psipR_(gp), 
        psipZ_(gp),
        ipol_(gp),
        R_0(gp.R_0){ }

    /**
    * @brief \f[ \frac{ I_{pol}(R,Z)}{R \sqrt{\nabla\psi_p}} \f]
    */
    double operator()( double R, double Z) const
    {
        return (1./R)*(ipol_(R,Z)/sqrt(psipR_(R,Z)*psipR_(R,Z) +psipZ_(R,Z)*psipZ_(R,Z))) ;
    }
    /**
     * @brief == operator()(R,Z)
     */ 
    double operator()( double R, double Z, double phi) const
    {
        return operator()(R,Z);
    }
    private:
    PsipR  psipR_;
    PsipZ  psipZ_;
    Ipol   ipol_;
    double R_0;  
};
///@} 
} //namespace dg