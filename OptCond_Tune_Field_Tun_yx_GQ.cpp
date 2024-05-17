/*************************************************************/
/* PROBLEM: (Nonlinear) magneto-optical response in TI films */
/* METHOD: Keldysh-Floquet Green's function formalism ********/
/* PROGRAMMER: Dr. Woo-Ram Lee (wlee51@ua.edu) ***************/
/*************************************************************/

/* DECLARATION: standard library */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

/* DECLARATION: MPI library*/

#include <mpi.h>

/* DECLARATION: GSL library */

#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>

#include <gsl/gsl_sf_bessel.h>

#include <gsl/gsl_rng.h>

#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>

#include <gsl/gsl_integration.h>

#include <gsl/gsl_multifit.h>

/* INPUT: numerical constants */

int	const		Ncpu        = 250;              /* total number of CPUs */
int const       MCC         = 1000/Ncpu;        /* number of function calls for Monte Carlo integration (2000000/Ncpu, 10000/Ncpu) */

int const       Fieldcut    = 101;              /* grid number of electric field strength (101) */
double const    dField      = 0.01;             /* grid size of electric field strength */
double const    Field0      = 0.00;             /* initial value inside the window for electric field */

double const    Domega      = 1.0/675;          /* grid size of frequency domain */
int const       Fmax        = 405;              /* grid number within unit frequency */
int const       N0cut       = 7;                /* cutoff number of Floquet mode */

/* INPUT: physical constants (in the unit of E_c) */

double const	Temp        = 0.02;             /* temperature (0.02, 0.14, 0.18, 0.22) */
double const	Eta         = 0.01;             /* level broadening width (0.01) */
double const    Ec          = 10.0;             /* Dirac band cutoff (10.0)*/
double const	Freq        = Fmax*Domega;      /* optical frequency */

/* INPUT: parameters for function */

struct gsl_params
{
    int rank, k, m, n;
    double Field, omega;
    double Coeff_Fit_0, Coeff_Fit_1, Coeff_Fit_2, Coeff_Fit_3, Coeff_Fit_4;
};


/*******************************************************************************/
/* SUBROUTINE: least-squares fitting to find the cutoff number of Floquet mode */
/*******************************************************************************/

double *Subroutine_Fit (double Field)
{
    /* DECLARATION: variables and arrays */
    
    int i, n, p, q, r, sgn, n_cut;
    double Ek, chisq;
    
    gsl_complex z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, Gr0;
    gsl_matrix *X, *cov;
    gsl_vector *Y, *c;
    
    static double Coeff_Fit[5];
    
    /* INITIALIZATION: least-squares fitting */
    
    X = gsl_matrix_alloc (20, 5);
    Y = gsl_vector_alloc (20);
    c = gsl_vector_alloc (5);
    cov = gsl_matrix_alloc (5, 5);
    
    /* LOOP: energy dispersion*/
    
    for (i = 0; i <= 19; i++)
    {
        Ek = 0.5 + i * Ec / 20.0;
        
        r = 0;
        
        /* LOOP: Floquet mode */
        
        for (n = (int)(Ek/Freq); n <= 1000000; n++)
        {
            /* INITIALIZATION: inverse retarded Green's function */
            
            gsl_matrix_complex *InvGr1 = gsl_matrix_complex_alloc (2*n+1, 2*n+1);
            
            /* LOOP: Floquet mode */
            
            for (p = 0; p <= 2*n; p++)
            {
                /* LOOP: Floquet mode */
                
                for (q = 0; q <= 2*n; q++)
                {
                    /* INPUT: inverse retarded Green's function */
                    
                    if (p == q) {
                        
                        GSL_SET_COMPLEX (&z1, (p-n)*Freq + 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, (p-n)*Freq - 0.5, Eta);
                        GSL_SET_COMPLEX (&z3, (p+1-n)*Freq - 0.5, Eta);
                        GSL_SET_COMPLEX (&z4, (p-1-n)*Freq - 0.5, Eta);
                        
                        GSL_SET_COMPLEX (&z5, pow(Ek, 2) - pow(0.5, 2), 0);
                        GSL_SET_COMPLEX (&z6, pow(0.5 * Field / Freq, 2), 0);
                        
                        z7 = gsl_complex_sub (gsl_complex_sub (gsl_complex_sub (z1, gsl_complex_div (z5, z2)), gsl_complex_div (z6, z3)), gsl_complex_div (z6, z4));
                        
                    } else if (p == q - 1) {
                        
                        GSL_SET_COMPLEX (&z1, (p-n)*Freq - 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, (p+1-n)*Freq - 0.5, Eta);
                        
                        GSL_SET_COMPLEX (&z3, 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * sin(0.5*M_PI),
                                         - 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * cos(0.5*M_PI));
                        GSL_SET_COMPLEX (&z4, - 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * sin(0.5*M_PI),
                                         - 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * cos(0.5*M_PI));
                        
                        z7 = gsl_complex_add (gsl_complex_div (z3, z1), gsl_complex_div (z4, z2));
                        
                    } else if (p == q + 1) {
                        
                        GSL_SET_COMPLEX (&z1, (p-n)*Freq - 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, (p-1-n)*Freq - 0.5, Eta);
                        
                        GSL_SET_COMPLEX (&z3, - 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * sin(0.5*M_PI),
                                         0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * cos(0.5*M_PI));
                        GSL_SET_COMPLEX (&z4, 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * sin(0.5*M_PI),
                                         0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * cos(0.5*M_PI));
                        
                        z7 = gsl_complex_add (gsl_complex_div (z3, z1), gsl_complex_div (z4, z2));
                        
                    } else if (p == q - 2) {
                        
                        GSL_SET_COMPLEX (&z1, (p+1-n)*Freq - 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, pow(0.5 * Field / Freq, 2), 0);
                        
                        z7 = gsl_complex_div (z2, z1);
                        
                    } else if (p == q + 2) {
                        
                        GSL_SET_COMPLEX (&z1, (p-1-n)*Freq - 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, pow(0.5 * Field / Freq, 2), 0);
                        
                        z7 = gsl_complex_div (z2, z1);
                        
                    } else {
                        
                        GSL_SET_COMPLEX (&z7, 0, 0);
                    }
                    
                    gsl_matrix_complex_set (InvGr1, p, q, z7);
                }
            }
            
            /* CALCULATION: matrix inversion to get the retarded Green's function */
            
            gsl_permutation *permGr1 = gsl_permutation_alloc (2*n+1);
            gsl_matrix_complex *Gr1 = gsl_matrix_complex_alloc (2*n+1, 2*n+1);
            gsl_linalg_complex_LU_decomp (InvGr1, permGr1, &sgn);
            gsl_linalg_complex_LU_invert (InvGr1, permGr1, Gr1);
            
            /* CALCULATION: retarded Green's function in equilibrium */
            
            GSL_SET_COMPLEX (&z8, n*Freq + 0.5, Eta);
            GSL_SET_COMPLEX (&z9, n*Freq - 0.5, Eta);
            GSL_SET_COMPLEX (&z10, pow(Ek, 2) - pow(0.5, 2), 0);
            
            Gr0 = gsl_complex_inverse (gsl_complex_sub (z8, gsl_complex_div (z10, z9)));
            
            /* CRITERION: matching with the equilibrium Green's function at the boundary - Part 1 */
            
            z11 = gsl_matrix_complex_get (Gr1, 2*n, 2*n);
            
            if (fabs((GSL_REAL (z11) - GSL_REAL (Gr0)) / GSL_REAL (Gr0)) < 0.1) {
                
                r += 1;
                
            } else {
                
                r = 0;
            }
            
            /* FREE: previous allocation for arrays */
            
            gsl_permutation_free (permGr1);
            gsl_matrix_complex_free (InvGr1);
            gsl_matrix_complex_free (Gr1);
            
            /* CRITERION: matching with the equilibrium Green's function at the boundary - Part 2 */
            
            if (r >= 3) {
                
                n_cut = 2*n;
                
                break;
            }
        }
        
        /* INPUT: arrays for least-squares fitting */
        
        gsl_matrix_set (X, i, 0, 1.0);
        gsl_matrix_set (X, i, 1, Ek);
        gsl_matrix_set (X, i, 2, Ek*Ek);
        gsl_matrix_set (X, i, 3, Ek*Ek*Ek);
        gsl_matrix_set (X, i, 4, Ek*Ek*Ek*Ek);
        gsl_vector_set (Y, i, n_cut);
    }
    
    /* CALCULATION: least-squares fitting */
    
    gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (20, 5);
    gsl_multifit_linear (X, Y, c, cov, &chisq, work);
    gsl_multifit_linear_free (work);
    
    for (i = 0; i <= 4; i++)
    {
        Coeff_Fit[i] = gsl_vector_get (c, i);
    }
    
    /* FREE: previous allocation for arrays */
    
    gsl_matrix_free (X);
    gsl_vector_free (Y);
    gsl_vector_free (c);
    gsl_matrix_free (cov);
    
    return Coeff_Fit;
}


/***************************************************/
/* SUBLOUTINE: Green's function                    */
/***************************************************/

double f_kGreen (double *vars, size_t dim, void *params)
{
    /* DECLARATION: variables for function g */
     
    /* vars[0]: variable for energy dispersion */
    /* vars[1]: variable for angle in momentum space */
    
    /* DECLARATION: parameters for function g1 */
    
    int rank = ((struct gsl_params *) params) -> rank;
    int k = ((struct gsl_params *) params) -> k;
    int m = ((struct gsl_params *) params) -> m;
    int n = ((struct gsl_params *) params) -> n;
    
    double Field = ((struct gsl_params *) params) -> Field;
    double omega = ((struct gsl_params *) params) -> omega;
    
    double Coeff_Fit_0 = ((struct gsl_params *) params) -> Coeff_Fit_0;
    double Coeff_Fit_1 = ((struct gsl_params *) params) -> Coeff_Fit_1;
    double Coeff_Fit_2 = ((struct gsl_params *) params) -> Coeff_Fit_2;
    double Coeff_Fit_3 = ((struct gsl_params *) params) -> Coeff_Fit_3;
    double Coeff_Fit_4 = ((struct gsl_params *) params) -> Coeff_Fit_4;
    
    /* DECLARATION: variables */
    
    int j, p, q, Ncut, sgn;
    double kGreen;
    
    gsl_complex z1, z2, z3, z4, z5, z6, z7;
    
    /* DETERMINATION: cutoff number of Floquet mode */
    
    Ncut = 1 + 2 * (Coeff_Fit_0 + Coeff_Fit_1 * vars[0] + Coeff_Fit_2 * pow(vars[0],2) + Coeff_Fit_3 * pow(vars[0],3) + Coeff_Fit_4 * pow(vars[0],4));
    
    if (m - n + (Ncut-1)/2 >= 0 && m - n + (Ncut-1)/2 <= Ncut-1 && m + n + (Ncut-1)/2 >= 0 && m + n + (Ncut-1)/2 <= Ncut-1) {
        
        /* INITIALIZATION: inverse retarded and advanced Green's functions */
        
        gsl_matrix_complex *InvGup = gsl_matrix_complex_alloc (Ncut, Ncut);
        gsl_matrix_complex *InvGdn = gsl_matrix_complex_alloc (Ncut, Ncut);
        
        /* LOOP: retarded Green's function (j=0 for spin up; j=1 for spin down) */
        
        for (j = 0; j <= 1; j++)
        {
            /* LOOP: Floquet mode */
            
            for (p = 0; p <= Ncut-1; p++)
            {
                /* LOOP: Floquet mode */
                
                for (q = 0; q <= Ncut-1; q++)
                {
                    /* INPUT: inverse retarded Green's function */
                    
                    if (p == q) {
                        
                        GSL_SET_COMPLEX (&z1, omega + (p-(Ncut-1)/2)*Freq + (2*j-1) * 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, omega + (p-(Ncut-1)/2)*Freq - (2*j-1) * 0.5, Eta);
                        GSL_SET_COMPLEX (&z3, omega + (p+1-(Ncut-1)/2)*Freq - (2*j-1) * 0.5, Eta);
                        GSL_SET_COMPLEX (&z4, omega + (p-1-(Ncut-1)/2)*Freq - (2*j-1) * 0.5, Eta);
                        
                        GSL_SET_COMPLEX (&z5, pow(vars[0], 2) - pow(0.5, 2), 0);
                        GSL_SET_COMPLEX (&z6, pow(0.5 * Field / Freq, 2), 0);
                        
                        z7 = gsl_complex_sub (gsl_complex_sub (gsl_complex_sub (z1, gsl_complex_div (z5, z2)), gsl_complex_div (z6, z3)), gsl_complex_div (z6, z4));
                        
                    } else if (p == q - 1) {
                        
                        GSL_SET_COMPLEX (&z1, omega + (p-(Ncut-1)/2)*Freq - (2*j-1) * 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, omega + (p+1-(Ncut-1)/2)*Freq - (2*j-1) * 0.5, Eta);
                        
                        GSL_SET_COMPLEX (&z3, (2*j-1) * 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]),
                                         - 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]));
                        GSL_SET_COMPLEX (&z4, - (2*j-1) * 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]),
                                         - 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]));
                        
                        z7 = gsl_complex_add (gsl_complex_div (z3, z1), gsl_complex_div (z4, z2));
                        
                    } else if (p == q + 1) {
                        
                        GSL_SET_COMPLEX (&z1, omega + (p-(Ncut-1)/2)*Freq - (2*j-1) * 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, omega + (p-1-(Ncut-1)/2)*Freq - (2*j-1) * 0.5, Eta);
                        
                        GSL_SET_COMPLEX (&z3, - (2*j-1) * 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]),
                                         0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]));
                        GSL_SET_COMPLEX (&z4, (2*j-1) * 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]),
                                         0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]));
                        
                        z7 = gsl_complex_add (gsl_complex_div (z3, z1), gsl_complex_div (z4, z2));
                        
                    } else if (p == q - 2) {
                        
                        GSL_SET_COMPLEX (&z1, omega + (p+1-(Ncut-1)/2)*Freq - (2*j-1) * 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, pow(0.5 * Field / Freq, 2), 0);
                        
                        z7 = gsl_complex_div (z2, z1);
                        
                    } else if (p == q + 2) {
                        
                        GSL_SET_COMPLEX (&z1, omega + (p-1-(Ncut-1)/2)*Freq - (2*j-1) * 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, pow(0.5 * Field / Freq, 2), 0);
                        
                        z7 = gsl_complex_div (z2, z1);
                        
                    } else {
                        
                        GSL_SET_COMPLEX (&z7, 0, 0);
                    }
                    
                    if (j == 0)
                    {
                        gsl_matrix_complex_set (InvGup, p, q, z7);
                    }
                    
                    if (j == 1)
                    {
                        gsl_matrix_complex_set (InvGdn, p, q, z7);
                    }
                }
            }
        }
        
        /* CALCULATION: matrix inversion to get the retarded Green's function for spin up */
        
        gsl_permutation *permGup = gsl_permutation_alloc (Ncut);
        gsl_matrix_complex *Gup = gsl_matrix_complex_alloc (Ncut, Ncut);
        gsl_linalg_complex_LU_decomp (InvGup, permGup, &sgn);
        gsl_linalg_complex_LU_invert (InvGup, permGup, Gup);
        
        /* CALCULATION: matrix inversion to get the retarded Green's function for spin down */
        
        gsl_permutation *permGdn = gsl_permutation_alloc (Ncut);
        gsl_matrix_complex *Gdn = gsl_matrix_complex_alloc (Ncut, Ncut);
        gsl_linalg_complex_LU_decomp (InvGdn, permGdn, &sgn);
        gsl_linalg_complex_LU_invert (InvGdn, permGdn, Gdn);
        
        z1 = gsl_matrix_complex_get (Gup, m - n + (Ncut-1)/2, m - n + (Ncut-1)/2);
        z2 = gsl_matrix_complex_get (Gdn, m - n + (Ncut-1)/2, m - n + (Ncut-1)/2);
        z3 = gsl_matrix_complex_get (Gup, m + n + (Ncut-1)/2, m + n + (Ncut-1)/2);
        z4 = gsl_matrix_complex_get (Gdn, m + n + (Ncut-1)/2, m + n + (Ncut-1)/2);
        
        if (k == 1) {
            
            kGreen = vars[0] / pow(2.0*M_PI,2) * (GSL_REAL (z1) + GSL_REAL (z2) + GSL_REAL (z3) + GSL_REAL (z4)) / Ncpu;
        }
        
        if (k == 2) {
            
            kGreen = vars[0] / pow(2.0*M_PI,2) * (GSL_IMAG (z1) + GSL_IMAG (z2)) / Ncpu;
        }
        
        /* FREE: previous allocation for arrays */
        
        gsl_permutation_free (permGup);
        gsl_matrix_complex_free (InvGup);
        gsl_matrix_complex_free (Gup);
        
        gsl_permutation_free (permGdn);
        gsl_matrix_complex_free (InvGdn);
        gsl_matrix_complex_free (Gdn);
        
    } else {
        
        kGreen = 0;
    }
    
    return kGreen;
}


/******************************************************/
/* SUBLOUTINE: local Green's function                 */
/******************************************************/

double Subroutine_LGreen (int rank, int k, int m, int n, double Field, double omega)
{
    /* DECLARATION: variables and arrays */
    
    double xl[2] = {0.5, 0};
    double xu[2] = {Ec, 2.0*M_PI};
    double *Coeff_Fit;
    double LGreen_Sub, error;
    
    /* DETERMINATION: cutoff number of Floquet mode */
    
    Coeff_Fit = Subroutine_Fit (Field);
    
    /* INTEGRATION: Monte Carlo Method (MISER) */
    
    struct gsl_params params_n = { rank, k, m, n, Field, omega, Coeff_Fit[0], Coeff_Fit[1], Coeff_Fit[2], Coeff_Fit[3], Coeff_Fit[4] };
    gsl_monte_function F_kGreen = { &f_kGreen, 2, &params_n };
    
    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup ();
    
    size_t calls = MCC;
    
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    gsl_rng_set (r, rank + 1);
    
    {
        gsl_monte_miser_state *s = gsl_monte_miser_alloc (2);
        gsl_monte_miser_integrate (&F_kGreen, xl, xu, 2, calls, r, s, &LGreen_Sub, &error);
        gsl_monte_miser_free (s);
    }
    
    gsl_rng_free (r);
    
    return LGreen_Sub;
}


/***************************************************/
/* SUBLOUTINE: Integrand for Qn                    */
/***************************************************/

double f_Qn (double x, void *params)
{
    /* DECLARATION: parameters for function */
    
    int rank = ((struct gsl_params *) params) -> rank;
    int k = ((struct gsl_params *) params) -> k;
    int m = ((struct gsl_params *) params) -> m;
    int n = ((struct gsl_params *) params) -> n;
    
    double Field = ((struct gsl_params *) params) -> Field;
    double omega = ((struct gsl_params *) params) -> omega;
    
    double Coeff_Fit_0 = ((struct gsl_params *) params) -> Coeff_Fit_0;
    double Coeff_Fit_1 = ((struct gsl_params *) params) -> Coeff_Fit_1;
    double Coeff_Fit_2 = ((struct gsl_params *) params) -> Coeff_Fit_2;
    double Coeff_Fit_3 = ((struct gsl_params *) params) -> Coeff_Fit_3;
    double Coeff_Fit_4 = ((struct gsl_params *) params) -> Coeff_Fit_4;
    
    /* DECLARATION: variables */
    
    int m0;
    double omega0;
    double LGreen_Sub, LGreenRe_n, LGreenIm_0, Qn_Int;
    
    /* Conversion: variables */
    
    m0 = (int) (x/Freq - 0.5);
    omega0 = x - m * Freq;
    
    /* PARALLELIZING: MPI */
    
    LGreen_Sub = Subroutine_LGreen (rank, 1, m0, n, Field, omega0);
    MPI_Allreduce(&LGreen_Sub, &LGreenRe_n, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    LGreen_Sub = Subroutine_LGreen (rank, 2, m0, 0, Field, omega0);
    MPI_Allreduce(&LGreen_Sub, &LGreenIm_0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    
    Qn_Int = - LGreenRe_n * (-LGreenIm_0/M_PI);     // Suppose a step function for the Fermi distribution with zero temperature
    
    return Qn_Int;
}


/****************/
/* MAIN ROUTINE */
/****************/

main(int argc, char **argv)
{
    /* OPEN: saving files */
    
    FILE *f1, *f2;
    f1 = fopen("LDOS_Ec10_T002_F060","wt");
    f2 = fopen("TunCond_Ec10_T002_F060","wt");
    
    /* INITIALIZATION: MPI */
    
    int rank;		/* Rank of my CPU */
    int dest = 0;	/* CPU receiving message */
    int tag = 0;	/* Indicator distinguishing messages */
    
    MPI_Status status;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    /* DECLARATION: variables and arrays */
    
    int m, n, r, iField, iomega;
    double Field, omega, Wc;
    double LGreen_Sub, LGreenIm_0, LGreenIm_Ref;
    double TunCond_Wn, TunCond_Qn, error;
/*
    // LOOP: Floquet mode
    
    for (m = -N0cut; m <= N0cut; m++) {
        
        // LOOP: frequency
        
        for (iomega = 0; iomega <= Fmax-1; iomega++)
        {
            omega = (iomega-(Fmax-1.0)/2.0)*Domega;
            
            // PARALLELIZING: MPI
            
            Field = 0.5;
 
            LGreen_Sub = Subroutine_LGreen (rank, 2, m, 0, Field, omega);
            MPI_Reduce(&LGreen_Sub, &LGreenIm_0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            
            // SAVE: data on LDOS
            
            if (rank == 0) {
                
                printf("%f %f %f %e\n", Freq, Field, omega + m * Freq, -LGreenIm_0/M_PI);
                fprintf(f1, "%f %f %f %e\n", Freq, Field, omega + m * Freq, -LGreenIm_0/M_PI);
            }
        }
    }
 
    if (rank == 0) {
        
        printf("\n");
        fprintf(f1, "\n");
    }
*/
    /* LOOP: Floquet mode */
    
    for (n = 0; n <= N0cut; n++)
    {
        /* LOOP: electric field */
        
        for (iField = 0; iField <= Fieldcut-1; iField++)
        {
            if ((n == 1 && Field >= 0.74) || n >= 2) {
             
                /* PRESCRIPTION: zero field */
                
                if (iField == 0) {
                    
                    Field = 0.0001;
                    
                } else {
                    
                    Field = Field0 + iField * dField;
                }
                
                /* CALCULATION: Tunneling conductivity */
                
                if (n == 0) {
                    
                    TunCond_Wn = gsl_sf_bessel_Jn(0,2.0*Field/pow(Freq,2)) * gsl_sf_bessel_Jn(1,2.0*Field/pow(Freq,2)) / Field;
                    
                } else {
                    
                    TunCond_Wn = gsl_sf_bessel_Jn(n,2.0*Field/pow(Freq,2)) * (gsl_sf_bessel_Jn(n+1,2.0*Field/pow(Freq,2)) - gsl_sf_bessel_Jn(n-1,2.0*Field/pow(Freq,2))) / Field;
                }
                
                /* CALCULATION: Finding the cutoff Wc */
                
                // PARALLELIZING: MPI
                
                r = 0;
                m = (int)(-Ec/Freq - 0.5);
                
                LGreen_Sub = Subroutine_LGreen (rank, 2, m, 0, Field, 0);
                MPI_Allreduce(&LGreen_Sub, &LGreenIm_Ref, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                
                // LOOP: Floquet mode
                
                for (m = (int)(-Ec/Freq - 0.5) - 1; m >= -1000000; m--)
                {
                    // PARALLELIZING: MPI
                    
                    LGreen_Sub = Subroutine_LGreen (rank, 2, m, 0, Field, 0);
                    MPI_Allreduce(&LGreen_Sub, &LGreenIm_0, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                    
                    // CRITERION: convergence test
                    
                    if (fabs(LGreenIm_0 / LGreenIm_Ref) < 1e-2) {
                        
                        r += 1;
                        
                    } else {
                        
                        r = 0;
                    }
                    
                    Wc = m * Freq;
                    
                    if (r >= 3) {
                        
                        break;
                    }
                }
                
                // INTEGRATION: QAGS adaptive routine
                
                struct gsl_params params_p = { rank, 0, 0, n, Field, 0, 0, 0, 0, 0, 0 };
                gsl_function F_Qn;
                F_Qn.function = &f_Qn;
                F_Qn.params = &params_p;
                
                gsl_integration_workspace *w = gsl_integration_workspace_alloc (1000);
                gsl_integration_qags (&F_Qn, Wc, 0, 0, 1e-2, 1000, w, &TunCond_Qn, &error);     // Suppose low temperature
                gsl_integration_workspace_free (w);
                
                // SAVE: data on tunneling conductivity
                
                if (rank == 0) {
                    
                    printf("%f %d %f %e %e %e\n", Freq, n, Field, TunCond_Wn, TunCond_Qn, TunCond_Wn * TunCond_Qn);
                    fprintf(f2, "%f %d %f %e %e %e\n", Freq, n, Field, TunCond_Wn, TunCond_Qn, TunCond_Wn * TunCond_Qn);
                }
            }
        }
        
        if (rank == 0) {
            
            printf("\n\n");
            fprintf(f2, "\n\n");
        }
    }
    
    /* CLOSE: saving files */
    
    fclose(f1);
    fclose(f2);
    
    /* FINISH: MPI */
    
    MPI_Finalize();
}

