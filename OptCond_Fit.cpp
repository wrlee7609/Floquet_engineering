/*************************************************************/
/* PROBLEM: (Nonlinear) magneto-optical response in TI films */
/* METHOD: Keldysh-Floquet Green's function formalism ********/
/* PROGRAMMER: Dr. Woo-Ram Lee (wlee51@ua.edu) ***************/
/*************************************************************/

/* DECLARATION: standard library */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>

#include <gsl/gsl_multifit.h>

/* INPUT: numerical constants */

int	const		Ncpu        = 10;				/* total number of CPUs */
int const       Fieldcut    = 5;                /* grid number of electric field strength */
double const    dField      = 0.5;              /* grid size of electric field strength */
double const    Field0      = 0;                /* initial value inside the window for electric field */

/* INPUT: physical constants (in the unit of E_c) */

double const	Eta         = 0.004;            /* level broadening width */
double const	Freq        = 0.5;              /* optical frequency */
double const    Ec          = 20.0;             /* Dirac band cutoff */


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
    
    static double Coeff_Fit[25];
    
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
            }
            
            /* FREE: previous allocation for arrays */
            
            gsl_permutation_free (permGr1);
            gsl_matrix_complex_free (InvGr1);
            gsl_matrix_complex_free (Gr1);
            
            /* CRITERION: matching with the equilibrium Green's function at the boundary - Part 2 */
            
            if (r >= 3) {
                
                n_cut = n;
                
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
        
        Coeff_Fit[i] = n_cut;
    }
    
    /* CALCULATION: least-squares fitting */
    
    gsl_multifit_linear_workspace * work = gsl_multifit_linear_alloc (20, 5);
    gsl_multifit_linear (X, Y, c, cov, &chisq, work);
    gsl_multifit_linear_free (work);
    
    for (i = 20; i <= 24; i++)
    {
        Coeff_Fit[i] = gsl_vector_get (c, i - 20);
    }
    
    /* FREE: previous allocation for arrays */
    
    gsl_matrix_free (X);
    gsl_vector_free (Y);
    gsl_vector_free (c);
    gsl_matrix_free (cov);
    
    return Coeff_Fit;
}


/****************/
/* MAIN ROUTINE */
/****************/

main(int argc, char **argv)
{
    /* OPEN: saving files */
    
    FILE *f1, *f2;
    f1 = fopen("OptCond_Fit_Main","wt");
    f2 = fopen("OptCond_Fit_Para","wt");
    
    /* INITIALIZATION: MPI */
    
    int rank;		/* Rank of my CPU */
    int source;		/* CPU sending message */
    int dest = 0;	/* CPU receiving message */
    int tag = 0;	/* Indicator distinguishing messages */
    
    MPI_Status status;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    /* DECLARATION: variables and arrays */
    
    int iField, i;
    double Field, Ek;
    double *Coeff_Fit;
    
    /* SAVE: data on the cutoff number of Floquet mode */
    
    if (rank == 0) {
        
        /* LOOP: electric field */
        
        for (iField = 0; iField <= Fieldcut - 1; iField++)
        {
            Field = Field0 + iField * dField;
            
            Coeff_Fit = Subroutine_Fit (Field);
            
            for (i = 0; i <= 19; i++)
            {
                Ek = 0.5 + i * Ec / 20.0;
                
                fprintf(f1, "%f %f %e \n", Field, Ek, Coeff_Fit[i]);
            }
            
            fprintf(f1, "\n\n");
            
            fprintf(f2, "%f ", Field);
            
            for (i = 20; i <= 24; i++)
            {
                fprintf(f2, "%e ", Coeff_Fit[i]);
            }
            
            fprintf(f2, "\n");
            
            printf("Calculation was done.");
        }
    }
    
    /* CLOSE: saving files */
    
    fclose(f1);
    fclose(f2);
    
    /* FINISH: MPI */
    
    MPI_Finalize();
}
