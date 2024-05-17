/****************************************************************************************/
/******************* [System] 3D TI (Strong, BiSe, ky-kz)             *******************/
/******************* [Method] Floquet eigensolver					  *******************/
/******************* [Programmer] Dr. Woo-Ram Lee (wrlee@kias.re.kr)  *******************/
/****************************************************************************************/

// MPI routines for parallel computing

#include <mpi.h>

// Standard libraries

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// GSL libraries

#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>

// Define 4 orbitals

#define RE1(z,i)((z)[2*(i)])
#define IM1(z,i)((z)[2*(i)+1])
#define RE2(z,i,j)((z)[(i)][2*(j)])
#define IM2(z,i,j)((z)[(i)][2*(j)+1])
 
// Numerical constants

int	const		Ncpu    = 15;				// total number of CPUs
int const		Ncut    = 75;				// grid number of Floquet mode (75, 261)
int const       Tmax    = Ncpu * 31;        // grid number of Bloch * time, 55 * 9
int const       Ksub    = 31;               // sub grid number of k_perp
int const		Kcut    = Ncpu * Ksub;      // total grid number of k_perp

// Physical constants (in unit 4D1)

double const	A1		= 0.6;
double const	A2		= 0.5;
double const	B1		= 0.6;
double const	B2		= 0.3;
double const	D2		= 0.2;
double const    C       = 0;
double const	M		= -0.3;

int const		Fmax    = Ncpu * 7;			// Field/Domega
int const		Iband   = Ncpu * 45;		// grid number of frequency domain in a bandwidth
double const	Bloch   = 1.0*Fmax/Iband;	// Bloch frequency
double const    Kplane  = 0.99 * M_PI;       // perpendicular momentum to the 2D subsystems
                                            // 0.0, 0.1, 0.2, 0.4, 0.6, 1.0

// Define subroutine

double (*SubSpectrum (int rank))[2];


/************************************************************************************************/
/************************************** [Main routine] ******************************************/
/************************************************************************************************/

main(int argc, char **argv)
{
    // Open the saving file
    
    FILE *f1;
    f1 = fopen("BiSe_Zak_Ey_16_kx_10","wt");
    
    // Initiate the MPI system
    
    int rank;		// Rank of my CPU
    int source;		// CPU sending message
    int dest = 0;	// CPU receiving message
    int tag = 0;	// Indicator distinguishing messages
    
    MPI_Status status;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    // Declare variables
    
    int ik, p;
    double (*SpectrumSub)[2];
    double Spectrum[Kcut][2];
    
    // Find the energy spectrum
    
    if (rank >= 1 && rank <= Ncpu-1) {
        
        SpectrumSub = SubSpectrum (rank);
        MPI_Send (SpectrumSub, Ksub*2, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        
    } else if (rank == 0) {
        
        for (source = 0; source <= Ncpu-1; source++)
        {
            if (source == 0) {
                
                SpectrumSub = SubSpectrum (source);
                
            } else {
                
                MPI_Recv (SpectrumSub, Ksub*2, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
            }
            
            for (ik = source*Ksub; ik <= (source+1)*Ksub-1; ik++)
            {
                for (p = 0; p <= 1; p++)
                {
                    Spectrum[ik][p] = SpectrumSub[ik-source*Ksub][p];
                }
            }
        }
    }
    
    // Save data on output files
    
    if (rank == 0) {
        
        for (ik = 0; ik <= Kcut-1; ik++)
        {
            fprintf(f1, "%f ", M_PI*(ik-(Kcut-1.0)/2.0)*2.0/Kcut);
            
            for (p = -2; p <= 2; p++)
            {
                fprintf(f1, "%f %f ", Spectrum[ik][1] + p, Spectrum[ik][0] + p+1);
            }
            
            fprintf(f1, "\n");
        }
        
        printf("Calculation is done.");
    }
    
    // Close save file
    
    fclose(f1);
    
    // Finish MPI system
    
    MPI_Finalize();
}


/************************************************************************************************/
/*************************** [Subroutine] Floquet Energy Spectrum *******************************/
/************************************************************************************************/

double (*SubSpectrum (int rank))[2]
{
    // Definition of local variables
    
    int ik, it, p, q;
    double Otime, Kperp;
    double Ek, Ek1, Eav, d1, d2, d3, d4, d5, Dd1, Dd2, Dd4, dnorm, Alpha, BetaRe, BetaIm;
    double FHuuRe, FHuuIm, FHudRe, FHudIm, FHduRe, FHduIm, FHddRe, FHddIm;
    static double SpectrumSub[Ksub][2];
    
    // Find the energy spectrum
    
    for (ik = rank*Ksub; ik <= (rank+1)*Ksub-1; ik++)
    {
        // Input conserved momenta
        
        Kperp = M_PI*(ik-(Kcut-1.0)/2.0)*2.0/Kcut;
        
        // Construct the Floquet matrix
        
        gsl_matrix_complex *FloHam = gsl_matrix_complex_alloc (2*Ncut, 2*Ncut);
        
        for (p = 0; p <= Ncut-1; p++)
        {
            for (q = 0; q <= Ncut-1; q++)
            {
                FHuuRe = FHuuIm = 0;
                FHudRe = FHudIm = 0;
                FHduRe = FHduIm = 0;
                FHddRe = FHddIm = 0;
                
                for (it = 0; it <= Tmax-1; it++)
                {
                    Otime = 2.0*M_PI * it/Tmax;
                    
                    Ek1 = C + 0.5 * (2.0 - cos(Kplane) - cos(Otime)) + 2.0 * D2 * (1.0 - cos(Kperp));
                    
                    d1 = A1 * sin(Kplane);
                    d2 = A1 * sin(Otime);
                    d3 = M + 2.0 * B1 * (2.0 - cos(Kplane) - cos(Otime)) + 2.0 * B2 * (1.0 - cos(Kperp));
                    d4 = A2 * sin(Kperp);
                    d5 = 0;
                    
                    Dd1 = A1 * cos(Kplane);
                    Dd2 = A1 * cos(Otime);
                    Dd4 = A2 * cos(Kperp);
                    
                    dnorm = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2) + pow(d4,2) + pow(d5,2));
                    
                    Alpha = (pow(dnorm - d3,2) + pow(d4,2)) / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
                    
                    FHuuRe += 1.0/(2.0*M_PI*Tmax) * cos((p-q)*Otime) * (Ek1 - dnorm + Bloch * Alpha * d1 * Dd2);
                    FHuuIm += 1.0/(2.0*M_PI*Tmax) * sin((p-q)*Otime) * (Ek1 - dnorm + Bloch * Alpha * d1 * Dd2);
                    
                    FHddRe += 1.0/(2.0*M_PI*Tmax) * cos((p-q)*Otime) * (Ek1 - dnorm - Bloch * Alpha * d1 * Dd2);
                    FHddIm += 1.0/(2.0*M_PI*Tmax) * sin((p-q)*Otime) * (Ek1 - dnorm - Bloch * Alpha * d1 * Dd2);
                    
                    if (Kplane == 0 || Kplane == 1.0 * M_PI) {
                        
                        BetaRe = 1.0 / (2.0*dnorm*(dnorm - d3));
                        BetaIm = 0;
                        
                    } else {
                        
                        BetaRe = (- pow(d1,2) + pow(d2,2)) / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
                        BetaIm = - 2.0*d1*d2 / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
                    }
                    
                    FHudRe += 1.0/(2.0*M_PI*Tmax) * Bloch * (- BetaRe * cos((p-q)*Otime) + BetaIm * sin((p-q)*Otime)) * d4 * Dd2;
                    FHudIm += 1.0/(2.0*M_PI*Tmax) * Bloch * (- BetaIm * cos((p-q)*Otime) - BetaRe * sin((p-q)*Otime)) * d4 * Dd2;
                    
                    FHduRe += 1.0/(2.0*M_PI*Tmax) * Bloch * (- BetaRe * cos((p-q)*Otime) - BetaIm * sin((p-q)*Otime)) * d4 * Dd2;
                    FHduIm += 1.0/(2.0*M_PI*Tmax) * Bloch * (BetaIm * cos((p-q)*Otime) - BetaRe * sin((p-q)*Otime)) * d4 * Dd2;
                }
                
                if (p == q) {

                    FHuuRe = FHuuRe + (p-(Ncut-1.0)/2.0)*Bloch;
                    FHddRe = FHddRe + (p-(Ncut-1.0)/2.0)*Bloch;
                }
                
                gsl_matrix_complex_set (FloHam, 2*p, 2*q, gsl_complex_rect (FHuuRe, FHuuIm));
                gsl_matrix_complex_set (FloHam, 2*p, 2*q+1, gsl_complex_rect (FHudRe, FHudIm));
                gsl_matrix_complex_set (FloHam, 2*p+1, 2*q, gsl_complex_rect (FHduRe, FHduIm));
                gsl_matrix_complex_set (FloHam, 2*p+1, 2*q+1, gsl_complex_rect (FHddRe, FHddIm));
            }
        }
        
        // Floquet eigensolver
        
        gsl_vector *eval = gsl_vector_alloc (2*Ncut);
        gsl_matrix_complex *evec = gsl_matrix_complex_alloc (2*Ncut, 2*Ncut);
        gsl_eigen_hermv_workspace *w = gsl_eigen_hermv_alloc (2*Ncut);
        gsl_eigen_hermv (FloHam, eval, evec, w);
        gsl_eigen_hermv_sort (eval, evec, GSL_EIGEN_SORT_VAL_ASC);
        
        // Band center
        
        Eav = 0;
        
        for (it = 0; it <= Tmax-1; it++)
        {
            Otime = 2.0*M_PI * it/Tmax;
            
            Ek1 = C + 0.5 * (2.0 - cos(Kplane) - cos(Otime)) + 2.0 * D2 * (1.0 - cos(Kperp));
            
            d1 = A1 * sin(Kplane);
            d2 = A1 * sin(Otime);
            d3 = M + 2.0 * B1 * (2.0 - cos(Kplane) - cos(Otime)) + 2.0 * B2 * (1.0 - cos(Kperp));
            d4 = A2 * sin(Kperp);
            d5 = 0;
            
            dnorm = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2) + pow(d4,2) + pow(d5,2));
            
            Eav += 1.0/(2.0*M_PI*Tmax) * (Ek1 - dnorm);
        }
        
        // Energy spectrum
        
        Ek = gsl_vector_get (eval, Ncut-1);
        SpectrumSub[ik-rank*Ksub][0] = (Ek - Eav) * 2.0*M_PI / Bloch;
        
        Ek = gsl_vector_get (eval, Ncut);
        SpectrumSub[ik-rank*Ksub][1] = (Ek - Eav) * 2.0*M_PI / Bloch;
        
        // Free the previous allocation
        
        gsl_vector_free (eval);
        gsl_matrix_complex_free (evec);
        gsl_matrix_complex_free (FloHam);
        gsl_eigen_hermv_free (w);
    }
    
    return SpectrumSub;
}

