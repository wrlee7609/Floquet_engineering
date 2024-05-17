/**************************************************************/
/* SUBJECT: Dynamically tunable graphene tunneling transistor */
/* COMMENT: Undressed case                                    */
/* METHOD: Keldysh-Floquet Green's function formalism         */
/* PROGRAMMER: Dr. Woo-Ram Lee (wlee51@ua.edu)                */
/**************************************************************/

// DECLARATION: standard library

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// DECLARATION: MPI library

#include <mpi.h>

// DECLARATION: GSL library

#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_miser.h>

// INPUT: numerical constants (to be optimized)

int const	    Ncpu       = 100;               // total number of CPUs (250)
int const       MCC        = 100000/Ncpu;        // number of function calls for Monte Carlo integration (2000000/Ncpu)

// INPUT: electron sector (in the unit of Dirac mass)

double const    Ec         = 4.0;               // Dirac band cutoff (10.0)
double const    Eta        = 0.01;              // level broadening width (0.01)
double const    Temp       = 0.02;              // temperature (0.02)

// INPUT: tuning parameter

int const       Bias_cut   = 160;               // grid number of bias (101)
double const    Bias_diff  = 0.1;               // grid size of bias
double const    Bias_init  = 0.0;               // initial value inside the window for bias


/**********************************************************************/
/* SUBLOUTINE: Integrand in the inter-layer tunneling current formula */
/**********************************************************************/

struct gsl_params
{
    double Bias;
};

double f_Integrand (double *vars, size_t dim, void *params)
{
    // DECLARATION: variables for function
    
    // vars[0]: variable for frequency domain
    // vars[1]: variable for energy dispersion
    
    // DECLARATION: parameters for function
    
    double Bias = ((struct gsl_params *) params) -> Bias;
    
    // DECLARATION: variables

    double Mu_T, Mu_B;
    double DOS_Tc, DOS_Tv, Dist_T;
    double DOS_Bc, DOS_Bv, Dist_B;
    double Integrand;
    
    // ASSUMPTION: half-filling at each layer
    
    Mu_T = 0.5*Bias;
    Mu_B = -0.5*Bias;
    
    // CALCULATION: Spectral information on top layer
    
    DOS_Tc = - GSL_IMAG (gsl_complex_inverse (gsl_complex_rect (vars[0] - vars[1] - 0.5*Bias, Eta))) / M_PI;
    DOS_Tv = - GSL_IMAG (gsl_complex_inverse (gsl_complex_rect (vars[0] + vars[1] - 0.5*Bias, Eta))) / M_PI;
    Dist_T = 1.0 / (exp((vars[0] - Mu_T)/Temp) + 1.0);
    
    // CALCULATION: Spectral information on bottom layer
    
    DOS_Bc = - GSL_IMAG (gsl_complex_inverse (gsl_complex_rect (vars[0] - vars[1] + 0.5*Bias, Eta))) / M_PI;
    DOS_Bv = - GSL_IMAG (gsl_complex_inverse (gsl_complex_rect (vars[0] + vars[1] + 0.5*Bias, Eta))) / M_PI;
    Dist_B = 1.0 / (exp((vars[0] - Mu_B)/Temp) + 1.0);
    
    // CALCULATION: Integrand in the inter-layer tunneling current formula
    
    Integrand = M_PI * vars[1] * ((1.0 - 0.5/vars[1]) * DOS_Tc + (1.0 + 0.5/vars[1]) * DOS_Tv) * ((1.0 - 0.5/vars[1]) * DOS_Bc + (1.0 + 0.5/vars[1]) * DOS_Bv) * (Dist_T - Dist_B);
    
    return Integrand;
}


/*************************************************************/
/* SUBLOUTINE: Inter-layer tunneling current vs bias voltage */
/*************************************************************/

double Subroutine_TunCurrent (int rank, double Bias)
{
    // DECLARATION: variables and arrays
    
    double xl[2] = {-Ec-Bias-1.0, 0.5};
    double xu[2] = {Ec+Bias+1.0, Ec};
    double error;
    double TunCurrent_Sub;
    
    // INTEGRATION: Monte Carlo Method (MISER)
    
    struct gsl_params params_p = { Bias };
    gsl_monte_function F_Integrand = { &f_Integrand, 2, &params_p };
    
    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup ();
    
    size_t calls = MCC;
    
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    gsl_rng_set (r, rank + 1);
    
    {
        gsl_monte_miser_state *s = gsl_monte_miser_alloc (2);
        gsl_monte_miser_integrate (&F_Integrand, xl, xu, 2, calls, r, s, &TunCurrent_Sub, &error);
        gsl_monte_miser_free (s);
    }
    
    gsl_rng_free (r);

    return TunCurrent_Sub;
}


/********************************************************/
/*                      MAIN ROUTINE                    */
/********************************************************/

main(int argc, char **argv)
{
    // OPEN: saving files
    
    FILE *f1;
    f1 = fopen("TunCurrent_T0p02_Ec04p0_E0p00_Mc100000","wt");
    
    // INITIALIZATION: MPI
    
    int rank;       // Rank of my CPU
    int dest = 0;	// CPU receiving message
    int tag = 0;	// Indicator distinguishing messages
    
    MPI_Status status;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    // DECLARATION: variables and arrays
    
    int j;
    double Bias;
    double TunCurrent_Sub;
    double TunCurrent;
    
    // LOOP: Bias voltage
    
    for (j = 0; j <= Bias_cut - 1; j++)
    {
        Bias = Bias_init + j * Bias_diff;
        
        // PARALLELIZING: MPI
        
        TunCurrent_Sub = Subroutine_TunCurrent (rank, Bias);
        
        MPI_Reduce(&TunCurrent_Sub, &TunCurrent, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        // SAVE: {bias voltage, inter-layer tunneling current}
        
        if (rank == 0) {
            
            printf("%f %e\n", Bias, TunCurrent/Ncpu);
            fprintf(f1, "%f %e\n", Bias, TunCurrent/Ncpu);
        }
    }
    
    if (rank == 0) {
        
        printf("\n");
        fprintf(f1, "\n");
    }
    
    // CLOSE: saving files
    
    fclose(f1);
    
    // FINISH: MPI
    
    MPI_Finalize();
}

