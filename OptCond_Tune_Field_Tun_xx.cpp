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

#include <gsl/gsl_multifit.h>

/* INPUT: numerical constants */

int	const		Ncpu        = 250;              /* total number of CPUs */
int const       MCC         = 10000/Ncpu;       /* number of function calls for Monte Carlo integration (2000000/Ncpu) */

int const       Fieldcut    = 101;              /* grid number of electric field strength (101) */
double const    dField      = 0.01;             /* grid size of electric field strength */
double const    Field0      = 0.00;             /* initial value inside the window for electric field */

double const    Domega      = 1.0/675;          /* grid size of frequency domain */
int const       Fmax        = 270;              /* grid number within unit frequency */
int const       N0cut       = 7;                /* cutoff number of Floquet mode */

/* INPUT: physical constants (in the unit of E_c) */

double const	Temp        = 0.02;             /* temperature (0.02, 0.14, 0.18, 0.22) */
double const	Eta         = 0.01;             /* level broadening width (0.01) */
double const    Ec          = 10.0;             /* Dirac band cutoff (10.0)*/
double const	Freq        = Fmax*Domega;      /* optical frequency */


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
/* SUBLOUTINE: spectral density (kDOS)             */
/***************************************************/

struct gsl_params
{
    int n;
    double omega, Field;
    double Coeff_Fit_0, Coeff_Fit_1, Coeff_Fit_2, Coeff_Fit_3, Coeff_Fit_4;
};

double f_kDOS (double *vars, size_t dim, void *params)
{
    /* DECLARATION: variables for function g */
     
    /* vars[0]: variable for energy dispersion */
    /* vars[1]: variable for angle in momentum space */
    
    /* DECLARATION: parameters for function g1 */
    
    int n = ((struct gsl_params *) params) -> n;
    double omega = ((struct gsl_params *) params) -> omega;
    double Field = ((struct gsl_params *) params) -> Field;
    
    double Coeff_Fit_0 = ((struct gsl_params *) params) -> Coeff_Fit_0;
    double Coeff_Fit_1 = ((struct gsl_params *) params) -> Coeff_Fit_1;
    double Coeff_Fit_2 = ((struct gsl_params *) params) -> Coeff_Fit_2;
    double Coeff_Fit_3 = ((struct gsl_params *) params) -> Coeff_Fit_3;
    double Coeff_Fit_4 = ((struct gsl_params *) params) -> Coeff_Fit_4;
    
    /* PRESCRIPTION: zero field */
    
    if (Field == 0) {
        
        Field = 0.0001;
    }
    
    /* DECLARATION: variables */
    
    int j, p, q, Ncut, sgn;
    double kDOS;
    
    gsl_complex z1, z2, z3, z4, z5, z6, z7;
    
    /* DETERMINATION: cutoff number of Floquet mode */
    
    Ncut = 1 + 2 * (Coeff_Fit_0 + Coeff_Fit_1 * vars[0] + Coeff_Fit_2 * pow(vars[0],2) + Coeff_Fit_3 * pow(vars[0],3) + Coeff_Fit_4 * pow(vars[0],4));
    
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
    
    z1 = gsl_matrix_complex_get (Gup, n + (Ncut-1)/2, n + (Ncut-1)/2);
    z2 = gsl_matrix_complex_get (Gdn, n + (Ncut-1)/2, n + (Ncut-1)/2);
    
    kDOS = - vars[0] / pow(2.0*M_PI,2) * (GSL_IMAG (z1) + GSL_IMAG (z2)) / M_PI;
    
    /* FREE: previous allocation for arrays */
    
    gsl_permutation_free (permGup);
    gsl_matrix_complex_free (InvGup);
    gsl_matrix_complex_free (Gup);
    
    gsl_permutation_free (permGdn);
    gsl_matrix_complex_free (InvGdn);
    gsl_matrix_complex_free (Gdn);
    
    return kDOS;
}


/******************************************************/
/* SUBLOUTINE: local density of states (LDOS)         */
/******************************************************/

double Subroutine_LDOS (int rank, int n, double omega, double Field)
{
    /* DECLARATION: variables and arrays */
    
    double xl[2] = {0.5, 0};
    double xu[2] = {Ec, 2.0*M_PI};
    double *Coeff_Fit;
    double error;
    double LDOS_Sub;
    
    /* DETERMINATION: cutoff number of Floquet mode */
    
    Coeff_Fit = Subroutine_Fit (Field);
    
    /* INTEGRATION: Monte Carlo Method (MISER) */
    
    struct gsl_params params_n = { n, omega, Field, Coeff_Fit[0], Coeff_Fit[1], Coeff_Fit[2], Coeff_Fit[3], Coeff_Fit[4] };
    gsl_monte_function F_kDOS = { &f_kDOS, 2, &params_n };
    
    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup ();
    
    size_t calls = MCC;
    
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    gsl_rng_set (r, rank + 1);
    
    {
        gsl_monte_miser_state *s = gsl_monte_miser_alloc (2);
        gsl_monte_miser_integrate (&F_kDOS, xl, xu, 2, calls, r, s, &LDOS_Sub, &error);
        gsl_monte_miser_free (s);
    }
    
    gsl_rng_free (r);
    
    return LDOS_Sub;
}


/****************/
/* MAIN ROUTINE */
/****************/

main(int argc, char **argv)
{
    /* OPEN: saving files */
    
    FILE *f1, *f2;
    f1 = fopen("LDOS_Ec10_T002_F040","wt");
    f2 = fopen("TunCond_Ec10_T002_F040","wt");
    
    /* INITIALIZATION: MPI */
    
    int rank;		/* Rank of my CPU */
    int dest = 0;	/* CPU receiving message */
    int tag = 0;	/* Indicator distinguishing messages */
    
    MPI_Status status;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    /* DECLARATION: variables and arrays */
    
    int m, n, iField, iomega;
    double Field, omega;
    double LDOS_Sub;
    double LDOS_0, LDOS_n;
    double TunCond_Wn, TunCond_Qn;
    
    /* LOOP: electric field */
/*
    for (iField = 0; iField <= Fieldcut-1; iField++)
    {
        Field = Field0 + iField * dField;
   
        if (Field == 0.5) {
*/
            /* LOOP: Floquet mode */
/*
            for (n = -N0cut; n <= N0cut; n++) {
*/
                /* LOOP: frequency */
/*
                for (iomega = 0; iomega <= Fmax-1; iomega++)
                {
                    omega = (iomega-(Fmax-1.0)/2.0)*Domega;
*/
                    /* PARALLELIZING: MPI */
/*
                    LDOS_Sub = Subroutine_LDOS (rank, n, omega, Field);
                    
                    MPI_Reduce(&LDOS_Sub, &LDOS_0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
*/
                    /* SAVE: data on LDOS */
/*
                    if (rank == 0) {
                        
                        printf("%f %f %f %e\n", Freq, Field, omega + n * Freq, LDOS_0/Ncpu);
                        fprintf(f1, "%f %f %f %e\n", Freq, Field, omega + n * Freq, LDOS_0/Ncpu);
                    }
                }
            }
        }
        
        if (rank == 0) {
            
            printf("\n");
            fprintf(f1, "\n");
        }
    }
*/
    /* LOOP: Floquet mode */
    
    for (n = 3; n <= N0cut; n++)    // n = 1
    {
        /* LOOP: electric field */
        
        for (iField = 0; iField <= Fieldcut-1; iField++)
        {
            Field = Field0 + iField * dField;
         
            /* PRESCRIPTION: zero field */
            
            if (Field == 0) {
                
                Field = 0.0001;
            }
            
            if ((n == 4 && Field >= 0.93) || n > 4) {
                
                /* CALCULATION: Tunneling conductivity */
                
                TunCond_Wn = gsl_sf_bessel_Jn(n,2.0*Field/pow(Freq,2)) * (gsl_sf_bessel_Jn(n+1,2.0*Field/pow(Freq,2)) + gsl_sf_bessel_Jn(n-1,2.0*Field/pow(Freq,2))) / Field;
                
                TunCond_Qn = 0;
                
                /* LOOP: frequency */
                
                for (iomega = 0; iomega <= (Fmax-1)/2; iomega++)
                {
                    omega = (iomega-(Fmax-1.0)/2.0)*Domega;
                    
                    /* PARALLELIZING: MPI */
                    
                    LDOS_Sub = Subroutine_LDOS (rank, 0, omega, Field);
                    
                    MPI_Reduce(&LDOS_Sub, &LDOS_0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    
                    LDOS_Sub = Subroutine_LDOS (rank, n, omega, Field);
                    
                    MPI_Reduce(&LDOS_Sub, &LDOS_n, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    
                    TunCond_Qn += Domega * (LDOS_0/Ncpu) * (LDOS_n/Ncpu);
                }
                
                /* LOOP: frequency */
                
                for (iomega = (Fmax-1)/2; iomega <= Fmax-1; iomega++)
                {
                    omega = (iomega-(Fmax-1.0)/2.0)*Domega;
                    
                    /* PARALLELIZING: MPI */
                    
                    LDOS_Sub = Subroutine_LDOS (rank, -n, omega, Field);
                    
                    MPI_Reduce(&LDOS_Sub, &LDOS_0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    
                    LDOS_Sub = Subroutine_LDOS (rank, 0, omega, Field);
                    
                    MPI_Reduce(&LDOS_Sub, &LDOS_n, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                    
                    TunCond_Qn += Domega * (LDOS_0/Ncpu) * (LDOS_n/Ncpu);
                }
                
                /* LOOP: Floquet mode */
                
                if (n >= 2) {
                    
                    for (m = 1; m <= n-1; m++) {
                        
                        /* LOOP: frequency */
                        
                        for (iomega = 0; iomega <= Fmax-1; iomega++)
                        {
                            omega = (iomega-(Fmax-1.0)/2.0)*Domega;
                            
                            /* PARALLELIZING: MPI */
                            
                            LDOS_Sub = Subroutine_LDOS (rank, -m, omega, Field);
                            
                            MPI_Reduce(&LDOS_Sub, &LDOS_0, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                            
                            LDOS_Sub = Subroutine_LDOS (rank, m, omega, Field);
                            
                            MPI_Reduce(&LDOS_Sub, &LDOS_n, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                            
                            TunCond_Qn += Domega * (LDOS_0/Ncpu) * (LDOS_n/Ncpu);
                        }
                    }
                }
                
                /* SAVE: data on tunneling conductivity */
                
                if (rank == 0) {
                    
                    printf("%f %d %f %e %e %e\n", Freq, n, Field, TunCond_Wn, TunCond_Qn, TunCond_Wn * TunCond_Qn);
                    fprintf(f2, "%f %d %f %e %e %e\n", Freq, n, Field, TunCond_Wn, TunCond_Qn, TunCond_Wn * TunCond_Qn);
                }
            }
        }
        
        if (rank == 0) {
            
            printf("\n");
            fprintf(f2, "\n");
        }
    }
    
    /* CLOSE: saving files */
    
    fclose(f1);
    fclose(f2);
    
    /* FINISH: MPI */
    
    MPI_Finalize();
}

