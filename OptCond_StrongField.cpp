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

int	const		Ncpu        = 40;				/* total number of CPUs */
int const       Fieldcut    = Ncpu;             /* grid number of electric field strength */
double const    dField      = 0.01;             /* grid size of electric field strength */
double const    Field0      = 0;                /* initial value inside the window for electric field */

/* INPUT: physical constants (in the unit of E_c) */

double const	Temp        = 0.02;             /* temperature */
double const	Eta         = 0.01;             /* level broadening width */
double const    Ec          = 10.0;             /* Dirac band cutoff */
double const	Freq        = 1.00;             /* optical frequency (0.3, 0.5, 0.7, 0.9) */


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


/*************************************************/
/* SUBLOUTINE: integrand of optical conductivity */
/*************************************************/

struct gsl_params
{
    int comp, iField;
    double Coeff_Fit_0, Coeff_Fit_1, Coeff_Fit_2, Coeff_Fit_3, Coeff_Fit_4;
};

double g (double *vars, size_t dim, void *params)
{
    /* DECLARATION: variables for function g */
     
    /* vars[0]: variable for energy dispersion */
    /* vars[1]: variable for angle in momentum space */
    /* vars[2]: variable for frequency domain */
    
    /* DECLARATION: parameters for function g */
    
    int comp = ((struct gsl_params *) params) -> comp;
    int iField = ((struct gsl_params *) params) -> iField;
    double Coeff_Fit_0 = ((struct gsl_params *) params) -> Coeff_Fit_0;
    double Coeff_Fit_1 = ((struct gsl_params *) params) -> Coeff_Fit_1;
    double Coeff_Fit_2 = ((struct gsl_params *) params) -> Coeff_Fit_2;
    double Coeff_Fit_3 = ((struct gsl_params *) params) -> Coeff_Fit_3;
    double Coeff_Fit_4 = ((struct gsl_params *) params) -> Coeff_Fit_4;
    
    double Field = Field0 + iField * dField;
    
    /* DECLARATION: variables */
    
    int j, k, p, q, n, l, sgn, Ncut;
    double sgn_band, OptCond_Int;
    
    gsl_complex z1, z2, z3, z4, z5, z6, z7, z8, z9, z10, z11, z12, z13, z14, z15, z16, z17, z18, z19, z20, z21, z22;
    gsl_complex Weight_X, Weight_Y, LesG, LesGp, LesGn;
    
    /* DETERMINATION: cutoff number of Floquet mode */
    
    Ncut = 1 + 2 * (Coeff_Fit_0 + Coeff_Fit_1 * vars[0] + Coeff_Fit_2 * pow(vars[0],2) + Coeff_Fit_3 * pow(vars[0],3) + Coeff_Fit_4 * pow(vars[0],4));
    
    /* DECLARATION: arrays */
    
    double Occ[Ncut], Weight_Wa[Ncut], Weight_Wb[Ncut];
    
    /* INITIALIZATION: inverse retarded and advanced Green's functions */
    
    gsl_matrix_complex *InvGr = gsl_matrix_complex_alloc (Ncut, Ncut);
    gsl_matrix_complex *InvGa = gsl_matrix_complex_alloc (Ncut, Ncut);
    
    /* LOOP: retarded and advanced Green's functions */
    
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
                    
                    GSL_SET_COMPLEX (&z1, vars[2] + (p-(Ncut-1)/2)*Freq + 0.5, Eta);
                    
                } else if (p == q - 1) {
                    
                    GSL_SET_COMPLEX (&z1, vars[2] + (p-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    GSL_SET_COMPLEX (&z2, vars[2] + (p+1-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    
                    GSL_SET_COMPLEX (&z3, 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]),
                                     - 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]));
                    GSL_SET_COMPLEX (&z4, - 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]),
                                     - 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]));
                    
                    z7 = gsl_complex_add (gsl_complex_div (z3, z1), gsl_complex_div (z4, z2));
                    
                } else if (p == q + 1) {
                    
                    GSL_SET_COMPLEX (&z1, vars[2] + (p-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    GSL_SET_COMPLEX (&z2, vars[2] + (p-1-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    
                    GSL_SET_COMPLEX (&z3, - 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]),
                                     0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]));
                    GSL_SET_COMPLEX (&z4, 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]),
                                     0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]));
                    
                    z7 = gsl_complex_add (gsl_complex_div (z3, z1), gsl_complex_div (z4, z2));
                    
                } else {
                    
                    GSL_SET_COMPLEX (&z7, 0, 0);
                }
                
                if (j == 0)
                {
                    gsl_matrix_complex_set (InvGr, p, q, z7);
                }
                
                if (j == 1)
                {
                    gsl_matrix_complex_set (InvGa, p, q, z7);
                }
            }
        }
    }
    
    /* CALCULATION: matrix inversion to get the retarded Green's function */
    
    gsl_permutation *permGr = gsl_permutation_alloc (Ncut);
    gsl_matrix_complex *Gr = gsl_matrix_complex_alloc (Ncut, Ncut);
    gsl_linalg_complex_LU_decomp (InvGr, permGr, &sgn);
    gsl_linalg_complex_LU_invert (InvGr, permGr, Gr);
    
    OptCond_Int = - vars[0]/M_PI/Field * (GSL_REAL (LesGp) + GSL_REAL (LesGn));
    
    /* FREE: previous allocation for arrays */
    
    gsl_permutation_free (permGr);
    gsl_permutation_free (permGa);
    gsl_matrix_complex_free (InvGr);
    gsl_matrix_complex_free (InvGa);
    gsl_matrix_complex_free (Gr);
    gsl_matrix_complex_free (Ga);
    
    return OptCond_Int;
}


/******************************************************/
/* SUBLOUTINE: optical conductivity vs electric field */
/******************************************************/

double *Subroutine_OptCond (int rank)
{
    /* DECLARATION: variables and arrays */
    
    int comp, iField;
    double Field, result, error;
    double *Coeff_Fit;
    double xl[3] = {0.5, 0, -0.5*Freq};
    double xu[3] = {Ec, 2.0*M_PI, 0.5*Freq};
    static double OptCond_Sub[2*Fieldcut/Ncpu];
    
    /* LOOP: electric field */
    
    for (iField = rank*Fieldcut/Ncpu; iField <= (rank+1)*Fieldcut/Ncpu-1; iField++)
    {
        Field = Field0 + iField * dField;
        
        /* DETERMINATION: cutoff number of Floquet mode */
        
        Coeff_Fit = Subroutine_Fit (Field);

        /* LOOP: component of optical conductivity */
        
        for (comp = 0; comp <= 1; comp++)
        {
            /* INTEGRATION: Monte Carlo Method (PLAIN / MISER / VEGAS) */
            
            struct gsl_params params_p = { comp, iField, Coeff_Fit[0], Coeff_Fit[1], Coeff_Fit[2], Coeff_Fit[3], Coeff_Fit[4] };
            gsl_monte_function G = { &g, 3, &params_p };
            
            const gsl_rng_type *T;
            gsl_rng *r;
            
            size_t calls = MCC;
            
            gsl_rng_env_setup ();
            
            T = gsl_rng_default;
            r = gsl_rng_alloc (T);
            
            {
                gsl_monte_miser_state *s = gsl_monte_miser_alloc (3);
                gsl_monte_miser_integrate (&G, xl, xu, 3, calls, r, s, &result, &error);
                gsl_monte_miser_free (s);
            }
            
            gsl_rng_free (r);
            
            OptCond_Sub[iField - rank * Fieldcut/Ncpu + comp * Fieldcut/Ncpu] = result;
            
            printf ("electric field = %f, comp = %d, result = %e, error = %e\n", Field, comp, result, error);
        }
    }
    
    return OptCond_Sub;
}


/****************/
/* MAIN ROUTINE */
/****************/

main(int argc, char **argv)
{
    /* OPEN: saving files */
    
    FILE *f1;
    f1 = fopen("OptCond_Ec10_T002_F100","wt");
    
    /* INITIALIZATION: MPI */
    
    int rank;		/* Rank of my CPU */
    int source;		/* CPU sending message */
    int dest = 0;	/* CPU receiving message */
    int tag = 0;	/* Indicator distinguishing messages */
    
    MPI_Status status;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    /* DECLARATION: variables and arrays */
    
    int comp, iField;
    double Field, HallAngle;
    double *OptCond_Sub;
    double OptCond[2*Fieldcut];
    
    /* PARALLELIZING: MPI */
    
    if (rank >= 1 && rank <= Ncpu-1) {
        
        OptCond_Sub = Subroutine_OptCond (rank);
        MPI_Send (OptCond_Sub, 2*Fieldcut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        
    } else if (rank == 0) {
        
        for (source = 0; source <= Ncpu-1; source++)
        {
            if (source == 0) {
                
                OptCond_Sub = Subroutine_OptCond (source);
                
            } else {
                
                MPI_Recv (OptCond_Sub, 2*Fieldcut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
            }
            
            /* LOOP: electric field */
            
            for (iField = source*Fieldcut/Ncpu; iField <= (source+1)*Fieldcut/Ncpu-1; iField++)
            {
                /* LOOP: component of optical conductivity */
                
                for (comp = 0; comp <= 1; comp++)
                {
                    OptCond[iField + comp * Fieldcut] = OptCond_Sub[iField - source * Fieldcut/Ncpu + comp * Fieldcut/Ncpu];
                }
            }
        }
    }
    
    /* SAVE: data on optical conductivity */
    
    if (rank == 0) {
        
        /* LOOP: electric field */
        
        for (iField = 0; iField <= Fieldcut-1; iField++)
        {
            Field = Field0 + iField * dField;
            HallAngle = atan(OptCond[iField + 1 * Fieldcut] / OptCond[iField + 0 * Fieldcut]);
            
            fprintf(f1, "%f %f %e %e %e\n",
                    Freq, Field,
                    OptCond[iField + 0 * Fieldcut],
                    OptCond[iField + 1 * Fieldcut],
                    HallAngle );

        }
        
        fprintf(f1, "\n");
        
        printf("Calculation was done.");
    }
    
    /* CLOSE: saving files */
    
    fclose(f1);
    
    /* FINISH: MPI */
    
    MPI_Finalize();
}

