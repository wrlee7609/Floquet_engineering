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
#include <gsl/gsl_math.h>
#include <gsl/gsl_integration.h>

// Numerical constants

int	const		Ncpu    = 15;                   // total number of CPUs
int const		Ncut    = 135;                  // cutoff number of Floquet mode
int const       Ksub    = 31;                   // sub grid number of k_perp
int const		Kcut    = Ncpu * Ksub;          // total grid number of k_perp

// Physical constants (in unit 4D1)

double const	A1		= 0.600000;
double const	A2		= 0.500000;
double const	B1		= 0.600000;
double const	B2		= 0.300000;
double const	D2		= 0.200000;
double const    C       = 0;
double const	M		= -0.300000;

int const		Fmax    = Ncpu * 7;             // Field/Domega
int const		Iband   = Ncpu * 45;            // grid number of frequency domain in a bandwidth
double const	Bloch   = 1.000000*Fmax/Iband;	// Bloch frequency
double const    Kplane  = 1.000000 * M_PI;      // perpendicular momentum to the 2D subsystems
                                                // 0.0, 0.1, 0.2, 0.4, 0.6, 1.0

// Define parameter set

struct gsl_params {
    
    double Kperp;
    int p, q;
};


/************************************************************************************************/
/****************************** [Subroutine] Floquet Hamiltonian ********************************/
/************************************************************************************************/

double FHuuRe (double x, void * params) {
    
    double Kperp = ((struct gsl_params *) params) -> Kperp;
    int p = ((struct gsl_params *) params) -> p;
    int q = ((struct gsl_params *) params) -> q;
    
    double Ek1 = C + 0.500000 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * D2 * (1.000000 - cos(Kperp));
    double d1 = A1 * sin(Kplane);
    double d2 = A1 * sin(x);
    double d3 = M + 2.000000 * B1 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * B2 * (1.000000 - cos(Kperp));
    double d4 = A2 * sin(Kperp);
    double d5 = 0;
    double Dd2 = A1 * cos(x);
    double dnorm = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2) + pow(d4,2) + pow(d5,2));
    double Alpha = (pow(dnorm - d3,2) + pow(d4,2)) / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
    
    double FHuuRe = cos((p-q)*x) * (Ek1 - dnorm + Alpha * Bloch * d1 * Dd2) / (2.000000*M_PI);
    return FHuuRe;
}

double FHuuIm (double x, void * params) {
    
    double Kperp = ((struct gsl_params *) params) -> Kperp;
    int p = ((struct gsl_params *) params) -> p;
    int q = ((struct gsl_params *) params) -> q;
    
    double Ek1 = C + 0.500000 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * D2 * (1.000000 - cos(Kperp));
    double d1 = A1 * sin(Kplane);
    double d2 = A1 * sin(x);
    double d3 = M + 2.000000 * B1 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * B2 * (1.000000 - cos(Kperp));
    double d4 = A2 * sin(Kperp);
    double d5 = 0;
    double Dd2 = A1 * cos(x);
    double dnorm = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2) + pow(d4,2) + pow(d5,2));
    double Alpha = (pow(dnorm - d3,2) + pow(d4,2)) / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
    
    double FHuuIm = sin((p-q)*x) * (Ek1 - dnorm + Alpha * Bloch * d1 * Dd2) / (2.000000*M_PI);
    return FHuuIm;
}

double FHddRe (double x, void * params) {
    
    double Kperp = ((struct gsl_params *) params) -> Kperp;
    int p = ((struct gsl_params *) params) -> p;
    int q = ((struct gsl_params *) params) -> q;
    
    double Ek1 = C + 0.500000 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * D2 * (1.000000 - cos(Kperp));
    double d1 = A1 * sin(Kplane);
    double d2 = A1 * sin(x);
    double d3 = M + 2.000000 * B1 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * B2 * (1.000000 - cos(Kperp));
    double d4 = A2 * sin(Kperp);
    double d5 = 0;
    double Dd2 = A1 * cos(x);
    double dnorm = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2) + pow(d4,2) + pow(d5,2));
    double Alpha = (pow(dnorm - d3,2) + pow(d4,2)) / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
    
    double FHddRe = cos((p-q)*x) * (Ek1 - dnorm - Alpha * Bloch * d1 * Dd2) / (2.000000*M_PI);
    return FHddRe;
}

double FHddIm (double x, void * params) {
    
    double Kperp = ((struct gsl_params *) params) -> Kperp;
    int p = ((struct gsl_params *) params) -> p;
    int q = ((struct gsl_params *) params) -> q;
    
    double Ek1 = C + 0.500000 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * D2 * (1.000000 - cos(Kperp));
    double d1 = A1 * sin(Kplane);
    double d2 = A1 * sin(x);
    double d3 = M + 2.000000 * B1 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * B2 * (1.000000 - cos(Kperp));
    double d4 = A2 * sin(Kperp);
    double d5 = 0;
    double Dd2 = A1 * cos(x);
    double dnorm = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2) + pow(d4,2) + pow(d5,2));
    double Alpha = (pow(dnorm - d3,2) + pow(d4,2)) / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
    
    double FHddIm = sin((p-q)*x) * (Ek1 - dnorm - Alpha * Bloch * d1 * Dd2) / (2.000000*M_PI);
    return FHddIm;
}

double FHudRe (double x, void * params) {
    
    double Kperp = ((struct gsl_params *) params) -> Kperp;
    int p = ((struct gsl_params *) params) -> p;
    int q = ((struct gsl_params *) params) -> q;
    
    double d1 = A1 * sin(Kplane);
    double d2 = A1 * sin(x);
    double d3 = M + 2.000000 * B1 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * B2 * (1.000000 - cos(Kperp));
    double d4 = A2 * sin(Kperp);
    double d5 = 0;
    double Dd2 = A1 * cos(x);
    double dnorm = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2) + pow(d4,2) + pow(d5,2));
    double BetaRe, BetaIm;
    
    if (Kplane == 0 || Kplane == 1.000000 * M_PI) {
        
        BetaRe = 1.0 / (2.0*dnorm*(dnorm - d3));
        BetaIm = 0;
        
    } else {
        
        BetaRe = (- pow(d1,2) + pow(d2,2)) / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
        BetaIm = - 2.0*d1*d2 / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
    }
    
    double FHudRe = (- cos((p-q)*x) * BetaRe + sin((p-q)*x) * BetaIm) * Bloch * d4 * Dd2 / (2.000000*M_PI);
    return FHudRe;
}

double FHudIm (double x, void * params) {
    
    double Kperp = ((struct gsl_params *) params) -> Kperp;
    int p = ((struct gsl_params *) params) -> p;
    int q = ((struct gsl_params *) params) -> q;
    
    double d1 = A1 * sin(Kplane);
    double d2 = A1 * sin(x);
    double d3 = M + 2.000000 * B1 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * B2 * (1.000000 - cos(Kperp));
    double d4 = A2 * sin(Kperp);
    double d5 = 0;
    double Dd2 = A1 * cos(x);
    double dnorm = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2) + pow(d4,2) + pow(d5,2));
    double BetaRe, BetaIm;
    
    if (Kplane == 0 || Kplane == 1.000000 * M_PI) {
        
        BetaRe = 1.0 / (2.0*dnorm*(dnorm - d3));
        BetaIm = 0;
        
    } else {
        
        BetaRe = (- pow(d1,2) + pow(d2,2)) / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
        BetaIm = - 2.0*d1*d2 / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
    }
    
    double FHudIm = (- cos((p-q)*x) * BetaIm - sin((p-q)*x) * BetaRe) * Bloch * d4 * Dd2 / (2.000000*M_PI);
    return FHudIm;
}

double FHduRe (double x, void * params) {
    
    double Kperp = ((struct gsl_params *) params) -> Kperp;
    int p = ((struct gsl_params *) params) -> p;
    int q = ((struct gsl_params *) params) -> q;
    
    double d1 = A1 * sin(Kplane);
    double d2 = A1 * sin(x);
    double d3 = M + 2.000000 * B1 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * B2 * (1.000000 - cos(Kperp));
    double d4 = A2 * sin(Kperp);
    double d5 = 0;
    double Dd2 = A1 * cos(x);
    double dnorm = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2) + pow(d4,2) + pow(d5,2));
    double BetaRe, BetaIm;
    
    if (Kplane == 0 || Kplane == 1.000000 * M_PI) {
        
        BetaRe = 1.0 / (2.0*dnorm*(dnorm - d3));
        BetaIm = 0;
        
    } else {
        
        BetaRe = (- pow(d1,2) + pow(d2,2)) / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
        BetaIm = - 2.0*d1*d2 / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
    }
    
    double FHduRe = (- cos((p-q)*x) * BetaRe - sin((p-q)*x) * BetaIm) * Bloch * d4 * Dd2 / (2.000000*M_PI);
    return FHduRe;
}

double FHduIm (double x, void * params) {
    
    double Kperp = ((struct gsl_params *) params) -> Kperp;
    int p = ((struct gsl_params *) params) -> p;
    int q = ((struct gsl_params *) params) -> q;
    
    double d1 = A1 * sin(Kplane);
    double d2 = A1 * sin(x);
    double d3 = M + 2.000000 * B1 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * B2 * (1.000000 - cos(Kperp));
    double d4 = A2 * sin(Kperp);
    double d5 = 0;
    double Dd2 = A1 * cos(x);
    double dnorm = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2) + pow(d4,2) + pow(d5,2));
    double BetaRe, BetaIm;
    
    if (Kplane == 0 || Kplane == 1.000000 * M_PI) {
        
        BetaRe = 1.0 / (2.0*dnorm*(dnorm - d3));
        BetaIm = 0;
        
    } else {
        
        BetaRe = (- pow(d1,2) + pow(d2,2)) / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
        BetaIm = - 2.0*d1*d2 / (2.0*dnorm*(dnorm - d3)*(pow(d1,2) + pow(d2,2)));
    }
    
    double FHduIm = (cos((p-q)*x) * BetaIm - sin((p-q)*x) * BetaRe) * Bloch * d4 * Dd2 / (2.000000*M_PI);
    return FHduIm;
}

double Eav (double x, void * params) {
    
    double Kperp = *(double *) params;
    
    double Ek1 = C + 0.500000 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * D2 * (1.000000 - cos(Kperp));
    double d1 = A1 * sin(Kplane);
    double d2 = A1 * sin(x);
    double d3 = M + 2.000000 * B1 * (2.000000 - cos(Kplane) - cos(x)) + 2.000000 * B2 * (1.000000 - cos(Kperp));
    double d4 = A2 * sin(Kperp);
    double d5 = 0;
    double dnorm = sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2) + pow(d4,2) + pow(d5,2));
    
    double Eav = (Ek1 - dnorm) / (2.000000*M_PI);
    return Eav;
}


/************************************************************************************************/
/*************************** [Subroutine] Floquet Energy Spectrum *******************************/
/************************************************************************************************/

double (*SubSpectrum (int rank))[2]
{
    // Definition of local variables
    
    int ik, p, q;
    double Kperp, Ek;
    double pts[3];
    double FHuuRe_Int, FHuuIm_Int, FHddRe_Int, FHddIm_Int, FHudRe_Int, FHudIm_Int, FHduRe_Int, FHduIm_Int, Eav_Int;
    double FHuuRe_Err, FHuuIm_Err, FHddRe_Err, FHddIm_Err, FHudRe_Err, FHudIm_Err, FHduRe_Err, FHduIm_Err, Eav_Err;
    static double SpectrumSub[Ksub][2];
    
    pts[0] = -M_PI;
    pts[1] = 0;
    pts[2] = M_PI;
    
    // Find the energy spectrum
    
    for (ik = rank*Ksub; ik <= (rank+1)*Ksub-1; ik++)
    {
        // Input conserved momenta
        
        Kperp = M_PI*(ik-(Kcut-1.000000)/2.000000)*2.000000/Kcut;
        
        // Construct the Floquet matrix
        
        gsl_matrix_complex *FloHam = gsl_matrix_complex_alloc (2*Ncut, 2*Ncut);
        
        for (p = 0; p <= Ncut-1; p++)
        {
            for (q = 0; q <= Ncut-1; q++)
            {
                struct gsl_params params_p = { Kperp, p, q };
                
                // Integration (FHuu, FHdd)
                
                gsl_function F1;
                F1.function = &FHuuRe;
                F1.params = &params_p;
                
                gsl_integration_workspace *w1 = gsl_integration_workspace_alloc (1000);
                gsl_integration_qagp (&F1, pts, 3, 0, 0.000001, 1000, w1, &FHuuRe_Int, &FHuuRe_Err);
                gsl_integration_workspace_free (w1);
                
                gsl_function F2;
                F2.function = &FHuuIm;
                F2.params = &params_p;
                
                gsl_integration_workspace *w2 = gsl_integration_workspace_alloc (1000);
                gsl_integration_qagp (&F2, pts, 3, 0, 0.000001, 1000, w2, &FHuuIm_Int, &FHuuIm_Err);
                gsl_integration_workspace_free (w2);
                
                gsl_function F3;
                F3.function = &FHddRe;
                F3.params = &params_p;
                
                gsl_integration_workspace *w3 = gsl_integration_workspace_alloc (1000);
                gsl_integration_qagp (&F3, pts, 3, 0, 0.000001, 1000, w3, &FHddRe_Int, &FHddRe_Err);
                gsl_integration_workspace_free (w3);
                
                gsl_function F4;
                F4.function = &FHddIm;
                F4.params = &params_p;
                
                gsl_integration_workspace *w4 = gsl_integration_workspace_alloc (1000);
                gsl_integration_qagp (&F4, pts, 3, 0, 0.000001, 1000, w4, &FHddIm_Int, &FHddIm_Err);
                gsl_integration_workspace_free (w4);
                
                if (p == q) {
                    
                    FHuuRe_Int += (p-(Ncut-1.000000)/2.000000)*Bloch;
                    FHddRe_Int += (p-(Ncut-1.000000)/2.000000)*Bloch;
                }
                
                // Integration (FHud, FHdu)
                
                gsl_function F5;
                F5.function = &FHudRe;
                F5.params = &params_p;
                
                gsl_integration_workspace *w5 = gsl_integration_workspace_alloc (1000);
                gsl_integration_qagp (&F5, pts, 3, 0, 0.000001, 1000, w5, &FHudRe_Int, &FHudRe_Err);
                gsl_integration_workspace_free (w5);
                
                gsl_function F6;
                F6.function = &FHudIm;
                F6.params = &params_p;
                
                gsl_integration_workspace *w6 = gsl_integration_workspace_alloc (1000);
                gsl_integration_qagp (&F6, pts, 3, 0, 0.000001, 1000, w6, &FHudIm_Int, &FHudIm_Err);
                gsl_integration_workspace_free (w6);
                
                gsl_function F7;
                F7.function = &FHduRe;
                F7.params = &params_p;
                
                gsl_integration_workspace *w7 = gsl_integration_workspace_alloc (1000);
                gsl_integration_qagp (&F7, pts, 3, 0, 0.000001, 1000, w7, &FHduRe_Int, &FHduRe_Err);
                gsl_integration_workspace_free (w7);
                
                gsl_function F8;
                F8.function = &FHduIm;
                F8.params = &params_p;
                
                gsl_integration_workspace *w8 = gsl_integration_workspace_alloc (1000);
                gsl_integration_qagp (&F8, pts, 3, 0, 0.000001, 1000, w8, &FHduIm_Int, &FHduIm_Err);
                gsl_integration_workspace_free (w8);
                
                // Set up Floquet matrix
                
                gsl_matrix_complex_set (FloHam, 2*p, 2*q, gsl_complex_rect (FHuuRe_Int, FHuuIm_Int));
                gsl_matrix_complex_set (FloHam, 2*p+1, 2*q+1, gsl_complex_rect (FHddRe_Int, FHddIm_Int));
                gsl_matrix_complex_set (FloHam, 2*p, 2*q+1, gsl_complex_rect (FHudRe_Int, FHudIm_Int));
                gsl_matrix_complex_set (FloHam, 2*p+1, 2*q, gsl_complex_rect (FHduRe_Int, FHduIm_Int));
            }
        }
        
        // Floquet eigensolver
        
        gsl_vector *eval = gsl_vector_alloc (2*Ncut);
        gsl_matrix_complex *evec = gsl_matrix_complex_alloc (2*Ncut, 2*Ncut);
        gsl_eigen_hermv_workspace *w = gsl_eigen_hermv_alloc (2*Ncut);
        
        gsl_eigen_hermv (FloHam, eval, evec, w);
        gsl_eigen_hermv_sort (eval, evec, GSL_EIGEN_SORT_VAL_ASC);
        
        gsl_matrix_complex_free (FloHam);
        gsl_matrix_complex_free (evec);
        gsl_eigen_hermv_free (w);
        
        // Band center
        
        gsl_function F9;
        F9.function = &Eav;
        F9.params = &Kperp;
        
        gsl_integration_workspace *w9 = gsl_integration_workspace_alloc (1000);
        gsl_integration_qagp (&F9, pts, 3, 0, 0.000001, 1000, w9, &Eav_Int, &Eav_Err);
        gsl_integration_workspace_free (w9);
        
        // Energy spectrum
        
        Ek = gsl_vector_get (eval, Ncut-1);
        SpectrumSub[ik-rank*Ksub][0] = (Ek - Eav_Int) / Bloch;
        
        Ek = gsl_vector_get (eval, Ncut);
        SpectrumSub[ik-rank*Ksub][1] = (Ek - Eav_Int) / Bloch;
        
        // Free the previous allocation
        
        gsl_vector_free (eval);
    }
    
    return SpectrumSub;
}


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
    
    gsl_set_error_handler_off();    // Turn off the error handler
    
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

