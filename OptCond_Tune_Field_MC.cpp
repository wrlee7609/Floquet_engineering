/*************************************************************/
/* PROBLEM: Dynamical QAHE in strong optical fields **********/
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

#include <gsl/gsl_rng.h>

#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>

#include <gsl/gsl_multifit.h>

/* INPUT: numerical constants */

int	const		Ncpu        = 250;              /* total number of CPUs */
int const       MCC         = 2000000/Ncpu;     /* number of function calls for Monte Carlo integration (2000000/Ncpu) */
int const       Fieldcut    = 101;              /* grid number of electric field strength (101) */
double const    dField      = 0.01;             /* grid size of electric field strength */
double const    Field0      = 0.00;             /* initial value inside the window for electric field */

/* INPUT: physical constants (in the unit of E_c) */

double const	Temp        = 0.22;             /* temperature (0.02, 0.14, 0.18, 0.22) */
double const	Eta         = 0.01;             /* level broadening width (0.01) */
double const    Ec          = 10.0;             /* Dirac band cutoff (10.0)*/
double const	Freq        = 0.20;             /* optical frequency */
int const       harmonics   = 1;                /* Fourier harmonics for dynamical conductivity */

double const    RWA1        = 1.0;
double const    RWA2        = 1.0;


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
                        
                        z7 = gsl_complex_sub (gsl_complex_sub (gsl_complex_sub (z1, gsl_complex_div (z5, z2)), gsl_complex_div (gsl_complex_mul_real (z6, pow(RWA2,2)), z3)), gsl_complex_div (gsl_complex_mul_real (z6, pow(RWA1,2)), z4));
                        
                    } else if (p == q - 1) {
                        
                        GSL_SET_COMPLEX (&z1, (p-n)*Freq - 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, (p+1-n)*Freq - 0.5, Eta);
                        
                        GSL_SET_COMPLEX (&z3, RWA1 * 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * sin(0.5*M_PI),
                                         - RWA1 * 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * cos(0.5*M_PI));
                        GSL_SET_COMPLEX (&z4, - RWA2 * 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * sin(0.5*M_PI),
                                         - RWA2 * 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * cos(0.5*M_PI));
                        
                        z7 = gsl_complex_add (gsl_complex_div (z3, z1), gsl_complex_div (z4, z2));
                        
                    } else if (p == q + 1) {
                        
                        GSL_SET_COMPLEX (&z1, (p-n)*Freq - 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, (p-1-n)*Freq - 0.5, Eta);
                        
                        GSL_SET_COMPLEX (&z3, - RWA2 * 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * sin(0.5*M_PI),
                                         RWA2 * 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * cos(0.5*M_PI));
                        GSL_SET_COMPLEX (&z4, RWA1 * 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * sin(0.5*M_PI),
                                         RWA1 * 0.5 * Field / Freq * sqrt(pow(Ek, 2) - pow(0.5, 2)) * cos(0.5*M_PI));
                        
                        z7 = gsl_complex_add (gsl_complex_div (z3, z1), gsl_complex_div (z4, z2));
                        
                    } else if (p == q - 2) {
                        
                        GSL_SET_COMPLEX (&z1, (p+1-n)*Freq - 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, RWA1 * RWA2 * pow(0.5 * Field / Freq, 2), 0);
                        
                        z7 = gsl_complex_div (z2, z1);
                        
                    } else if (p == q + 2) {
                        
                        GSL_SET_COMPLEX (&z1, (p-1-n)*Freq - 0.5, Eta);
                        GSL_SET_COMPLEX (&z2, RWA1 * RWA2 * pow(0.5 * Field / Freq, 2), 0);
                        
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
/* SUBLOUTINE: integrand of dynamical conductivity */
/***************************************************/

struct gsl_params
{
    int comp;
    double Field, Coeff_Fit_0, Coeff_Fit_1, Coeff_Fit_2, Coeff_Fit_3, Coeff_Fit_4;
};

double g (double *vars, size_t dim, void *params)
{
    /* DECLARATION: variables for function g */
     
    /* vars[0]: variable for energy dispersion */
    /* vars[1]: variable for angle in momentum space */
    /* vars[2]: variable for frequency domain */
    
    /* DECLARATION: parameters for function g */
    
    int comp = ((struct gsl_params *) params) -> comp;
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
                    
                    GSL_SET_COMPLEX (&z1, vars[2] + (p-(Ncut-1)/2)*Freq + 0.5, (-2*j+1)*Eta);
                    GSL_SET_COMPLEX (&z2, vars[2] + (p-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    GSL_SET_COMPLEX (&z3, vars[2] + (p+1-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    GSL_SET_COMPLEX (&z4, vars[2] + (p-1-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    
                    GSL_SET_COMPLEX (&z5, pow(vars[0], 2) - pow(0.5, 2), 0);
                    GSL_SET_COMPLEX (&z6, pow(0.5 * Field / Freq, 2), 0);
                    
                    z7 = gsl_complex_sub (gsl_complex_sub (gsl_complex_sub (z1, gsl_complex_div (z5, z2)), gsl_complex_div (gsl_complex_mul_real (z6, pow(RWA2,2)), z3)), gsl_complex_div (gsl_complex_mul_real (z6, pow(RWA1,2)), z4));
                    
                } else if (p == q - 1) {
                    
                    GSL_SET_COMPLEX (&z1, vars[2] + (p-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    GSL_SET_COMPLEX (&z2, vars[2] + (p+1-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    
                    GSL_SET_COMPLEX (&z3, RWA1 * 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]),
                                     - RWA1 * 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]));
                    GSL_SET_COMPLEX (&z4, - RWA2 * 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]),
                                     - RWA2 * 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]));
                    
                    z7 = gsl_complex_add (gsl_complex_div (z3, z1), gsl_complex_div (z4, z2));
                    
                } else if (p == q + 1) {
                    
                    GSL_SET_COMPLEX (&z1, vars[2] + (p-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    GSL_SET_COMPLEX (&z2, vars[2] + (p-1-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    
                    GSL_SET_COMPLEX (&z3, - RWA2 * 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]),
                                     RWA2 * 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]));
                    GSL_SET_COMPLEX (&z4, RWA1 * 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]),
                                     RWA1 * 0.5 * Field / Freq * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]));
                    
                    z7 = gsl_complex_add (gsl_complex_div (z3, z1), gsl_complex_div (z4, z2));
                    
                } else if (p == q - 2) {
                    
                    GSL_SET_COMPLEX (&z1, vars[2] + (p+1-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    GSL_SET_COMPLEX (&z2, RWA1 * RWA2 * pow(0.5 * Field / Freq, 2), 0);
                    
                    z7 = gsl_complex_div (z2, z1);
                    
                } else if (p == q + 2) {
                    
                    GSL_SET_COMPLEX (&z1, vars[2] + (p-1-(Ncut-1)/2)*Freq - 0.5, (-2*j+1)*Eta);
                    GSL_SET_COMPLEX (&z2, RWA1 * RWA2 * pow(0.5 * Field / Freq, 2), 0);
                    
                    z7 = gsl_complex_div (z2, z1);
                    
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
    
    /* CALCULATION: matrix inversion to get the advanced Green's function */
    
    gsl_permutation *permGa = gsl_permutation_alloc (Ncut);
    gsl_matrix_complex *Ga = gsl_matrix_complex_alloc (Ncut, Ncut);
    gsl_linalg_complex_LU_decomp (InvGa, permGa, &sgn);
    gsl_linalg_complex_LU_invert (InvGa, permGa, Ga);
    
    /* LOOP: emission & absorption processes */
    
    for (j = 0; j <= 1; j++)
    {
        /* INITIALIZATION: lesser Green's function */
        
        GSL_SET_COMPLEX (&LesG, 0, 0);
        
        /* LOOP: band index */
        
        for (k = 0; k <= 1; k++)
        {
            sgn_band = 2.0 * k - 1.0;
            
            /* LOOP: Floquet mode */
            
            for (p = 0; p <= Ncut-1; p++)
            {
                /* INPUT: electron occupation function in the band basis */
                
                Occ[p] = Eta / M_PI / (pow(vars[2] + (p-(Ncut-1)/2)*Freq - sgn_band * vars[0], 2) + pow(Eta, 2)) / (exp((vars[2] + (p-(Ncut-1)/2)*Freq) / Temp) + 1.0);
                
                /* INPUT: weight function Wa */
                
                Weight_Wa[p] = 0.5 * (1.0 + sgn_band * 0.5 / vars[0] - (1.0 - sgn_band * 0.5 / vars[0]) * (pow(vars[0], 2) - pow(0.5, 2))
                                / (pow(vars[2] + (p-(Ncut-1)/2)*Freq - 0.5, 2) + pow(Eta, 2)))
                                * (pow(vars[2] + (p-(Ncut-1)/2)*Freq - vars[0], 2) + pow(Eta, 2)) * (pow(vars[2] + (p-(Ncut-1)/2)*Freq + vars[0], 2) + pow(Eta, 2))
                                / ((pow(vars[2] + (p-(Ncut-1)/2)*Freq - 0.5, 2) + pow(Eta, 2)) * (pow(vars[2] + (p-(Ncut-1)/2)*Freq + 0.5, 2) + pow(Eta, 2))
                                - pow(pow(vars[0], 2) - pow(0.5, 2), 2));
                
                /* INPUT: weight function Wb */
                
                Weight_Wb[p] = 0.5 * (1.0 - sgn_band * 0.5 / vars[0] - (1.0 + sgn_band * 0.5 / vars[0]) * (pow(vars[0], 2) - pow(0.5, 2))
                                / (pow(vars[2] + (p-(Ncut-1)/2)*Freq + 0.5, 2) + pow(Eta, 2)))
                                * (pow(vars[2] + (p-(Ncut-1)/2)*Freq - vars[0], 2) + pow(Eta, 2)) * (pow(vars[2] + (p-(Ncut-1)/2)*Freq + vars[0], 2) + pow(Eta, 2))
                                / ((pow(vars[2] + (p-(Ncut-1)/2)*Freq - 0.5, 2) + pow(Eta, 2)) * (pow(vars[2] + (p-(Ncut-1)/2)*Freq + 0.5, 2) + pow(Eta, 2))
                                - pow(pow(vars[0], 2) - pow(0.5, 2), 2));
            }
            
            /* LOOP: Floquet mode */
            
            for (n = 1; n <= Ncut-2-harmonics; n++)
            {
                /* INPUT: lesser Green's function with the weight function X */
                
                GSL_SET_COMPLEX (&z1, Weight_Wa[n+(1-j)*harmonics] * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * cos(vars[1]),
                                 - Weight_Wa[n+(1-j)*harmonics] * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * sin(vars[1]));
                GSL_SET_COMPLEX (&z2, 0, Weight_Wa[n+(1-j)*harmonics] * 0.5 * Field / Freq);
                
                z3 = gsl_matrix_complex_get (Ga, n+(1-j)*harmonics, n+j*harmonics);
                z4 = gsl_complex_mul_real (gsl_matrix_complex_get (Ga, n+(1-j)*harmonics + 1, n+j*harmonics), RWA1);
                z5 = gsl_complex_mul_real (gsl_matrix_complex_get (Ga, n+(1-j)*harmonics - 1, n+j*harmonics), RWA2);
                
                Weight_X = gsl_complex_add (gsl_complex_mul (z1, z3), gsl_complex_mul (z2, gsl_complex_sub (z4, z5)));
                LesG = gsl_complex_add (LesG, gsl_complex_mul_imag (Weight_X, Occ[n+(1-j)*harmonics]));
                
                /* LOOP: Floquet mode */
                
                for (l = 1; l <= Ncut-2; l++)
                {
                    /* INPUT: lesser Green's function with the weight function Y */
                    
                    /* common factors */
                    
                    GSL_SET_COMPLEX (&z1, vars[2] + (n+(1-j)*harmonics-(Ncut-1)/2)*Freq - 0.5, Eta);
                    
                    z2 = gsl_matrix_complex_get (Gr, n+(1-j)*harmonics, l);
                    z3 = gsl_complex_sub (gsl_complex_mul_real (gsl_matrix_complex_get (Gr, n+(1-j)*harmonics + 1, l), RWA1), gsl_complex_mul_real (gsl_matrix_complex_get (Gr, n+(1-j)*harmonics - 1, l), RWA2));
                    z4 = gsl_complex_sub (gsl_complex_mul_real (gsl_matrix_complex_get (Gr, n+(1-j)*harmonics, l+1), RWA1), gsl_complex_mul_real (gsl_matrix_complex_get (Gr, n+(1-j)*harmonics, l-1), RWA2));
                    z5 = gsl_complex_sub (gsl_complex_sub (gsl_complex_mul_real (gsl_matrix_complex_get (Gr, n+(1-j)*harmonics + 1, l+1), pow(RWA1,2)), gsl_complex_mul_real (gsl_matrix_complex_get (Gr, n+(1-j)*harmonics + 1, l-1), RWA1 * RWA2)), gsl_complex_sub (gsl_complex_mul_real (gsl_matrix_complex_get (Gr, n+(1-j)*harmonics - 1, l+1), RWA1 * RWA2), gsl_complex_mul_real (gsl_matrix_complex_get (Gr, n+(1-j)*harmonics - 1, l-1), pow(RWA2,2))));
                    z6 = gsl_matrix_complex_get (Ga, l, n+j*harmonics);
                    z7 = gsl_complex_sub (gsl_complex_mul_real (gsl_matrix_complex_get (Ga, l+1, n+j*harmonics), RWA1), gsl_complex_mul_real (gsl_matrix_complex_get (Ga, l-1, n+j*harmonics), RWA2));
                    
                    /* 1st term */
                    
                    GSL_SET_COMPLEX (&z8, (pow(pow(vars[0], 2) - pow(0.5, 2), 1.5) * Weight_Wa[l] + sqrt(pow(vars[0], 2) - pow(0.5, 2))
                                        * (pow(vars[2] + (l-(Ncut-1)/2)*Freq + 0.5, 2) + pow(Eta, 2)) * Weight_Wb[l]) * cos(vars[1]),
                                        - (pow(pow(vars[0], 2) - pow(0.5, 2), 1.5) * Weight_Wa[l] + sqrt(pow(vars[0], 2) - pow(0.5, 2))
                                        * (pow(vars[2] + (l-(Ncut-1)/2)*Freq + 0.5, 2) + pow(Eta, 2)) * Weight_Wb[l]) * sin(vars[1]));
                    
                    z9 = gsl_complex_mul (gsl_complex_div (z8, z1), gsl_complex_mul (z2, z6));
                    
                    /* 2nd term */
                    
                    GSL_SET_COMPLEX (&z10, 0, 0.5 * Field / Freq);
                    GSL_SET_COMPLEX (&z11, cos(2.0*vars[1]), - sin(2.0*vars[1]));
                    
                    z12 = gsl_complex_mul_real (gsl_complex_add (gsl_complex_mul (z2, z7), gsl_complex_mul (gsl_complex_sub (z3, gsl_complex_mul (z11, z4)), z6)),
                                                (pow(vars[0], 2) - pow(0.5, 2)) * Weight_Wa[l]);
                    z13 = gsl_complex_mul_real (gsl_complex_mul (z3, z6), (pow(vars[2] + (l-(Ncut-1)/2)*Freq + 0.5, 2) + pow(Eta, 2)) * Weight_Wb[l]);
                    z14 = gsl_complex_mul (gsl_complex_div (z10, z1), gsl_complex_add (z12, z13));
                    
                    /* 3rd term */
                    
                    GSL_SET_COMPLEX (&z15, - pow(0.5 * Field / Freq, 2) * sqrt(pow(vars[0], 2) - pow(0.5, 2)) * Weight_Wa[l], 0);
                    GSL_SET_COMPLEX (&z16, cos(vars[1]), sin(vars[1]));
                    GSL_SET_COMPLEX (&z17, cos(vars[1]), - sin(vars[1]));
                    
                    z18 = gsl_complex_div (z15, z1);
                    z19 = gsl_complex_sub (gsl_complex_mul (gsl_complex_sub (gsl_complex_mul (z16, z3), gsl_complex_mul (z17, z4)), z7),
                                           gsl_complex_mul (z17, gsl_complex_mul (z5, z6)));
                    z20 = gsl_complex_mul (z18, z19);
                    
                    /* 4th term */
                    
                    GSL_SET_COMPLEX (&z21, 0, pow(0.5 * Field / Freq, 3) * Weight_Wa[l]);
                    
                    z22 = gsl_complex_mul (gsl_complex_div (z21, z1), gsl_complex_mul (z5, z7));
                    
                    /* sum up */
                    
                    Weight_Y = gsl_complex_add (gsl_complex_add (z9, z14), gsl_complex_add (z20, z22));
                    LesG = gsl_complex_add (LesG, gsl_complex_mul_imag (Weight_Y, Occ[l]));
                }
            }
        }
        
        if (j == 0)
        {
            GSL_SET_COMPLEX (&LesGp, GSL_REAL (LesG), GSL_IMAG (LesG));
        }
        
        if (j == 1)
        {
            GSL_SET_COMPLEX (&LesGn, GSL_REAL (LesG), GSL_IMAG (LesG));
        }
    }
    
    /* INPUT: integrand of dynamical conductivity */

    if (comp == 0) {    /* Re_OptCond_xx */

        OptCond_Int = - vars[0]/M_PI/Field * (GSL_IMAG (LesGp) + GSL_IMAG (LesGn));
    }

    if (comp == 1) {    /* Re_OptCond_yx */

        OptCond_Int = - vars[0]/M_PI/Field * (GSL_REAL (LesGp) + GSL_REAL (LesGn));
    }
    
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
/* SUBLOUTINE: dynamical conductivity vs electric field */
/******************************************************/

double Subroutine_OptCond (double Field, int comp, int rank)
{
    /* DECLARATION: variables and arrays */
    
    double xl[3] = {0.5, 0, -0.5*Freq};
    double xu[3] = {Ec, 2.0*M_PI, 0.5*Freq};
    double *Coeff_Fit;
    double error;
    double OptCond_Sub;
    
    /* DETERMINATION: cutoff number of Floquet mode */
    
    Coeff_Fit = Subroutine_Fit (Field);
    
    /* INTEGRATION: Monte Carlo Method (MISER) */
    
    struct gsl_params params_p = { comp, Field, Coeff_Fit[0], Coeff_Fit[1], Coeff_Fit[2], Coeff_Fit[3], Coeff_Fit[4] };
    gsl_monte_function G = { &g, 3, &params_p };
    
    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup ();
    
    size_t calls = MCC;
    
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);

    gsl_rng_set (r, rank + 1);
    
    {
        gsl_monte_miser_state *s = gsl_monte_miser_alloc (3);
        gsl_monte_miser_integrate (&G, xl, xu, 3, calls, r, s, &OptCond_Sub, &error);
        gsl_monte_miser_free (s);
    }
    
    gsl_rng_free (r);
/*
    if (rank == 0) {
     
        printf ("electric field = %f, comp = %d, result = %e, error = %e\n", Field, comp, OptCond_Sub, error);
    }
*/
    return OptCond_Sub;
}


/****************/
/* MAIN ROUTINE */
/****************/

main(int argc, char **argv)
{
    /* OPEN: saving files */
    
    FILE *f1;
    f1 = fopen("OptCond_Ec10_T022_F020","wt");
    
    /* INITIALIZATION: MPI */
    
    int rank;		/* Rank of my CPU */
    int dest = 0;	/* CPU receiving message */
    int tag = 0;	/* Indicator distinguishing messages */
    
    MPI_Status status;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    /* DECLARATION: variables and arrays */
    
    int comp, iField;
    double Field;
    double OptCond_xx, OptCond_yx;
    double OptCond_Sub;
    
    /* LOOP: electric field */
    
    for (iField = 0; iField <= Fieldcut-1; iField++)
    {
        Field = Field0 + iField * dField;
        
        /* LOOP: component of dynamical conductivity */
        
        for (comp = 0; comp <= 1; comp++)
        {
            /* PARALLELIZING: MPI */
            
            OptCond_Sub = Subroutine_OptCond (Field, comp, rank);
            
            if (comp == 0) {
                
                MPI_Reduce(&OptCond_Sub, &OptCond_xx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            }
            
            if (comp == 1) {
                
                MPI_Reduce(&OptCond_Sub, &OptCond_yx, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            }
        }
        
        /* SAVE: data on dynamical conductivity */
        
        if (rank == 0) {
            
            printf("%f %f %e %e %e\n", Freq, Field, OptCond_xx/Ncpu, OptCond_yx/Ncpu, atan(OptCond_yx/OptCond_xx));
            fprintf(f1, "%f %f %e %e %e\n", Freq, Field, OptCond_xx/Ncpu, OptCond_yx/Ncpu, atan(OptCond_yx/OptCond_xx));
        }
    }
    
    if (rank == 0) {
        
        printf("\n");
        fprintf(f1, "\n");
    }
    
    /* CLOSE: saving files */
    
    fclose(f1);
    
    /* FINISH: MPI */
    
    MPI_Finalize();
}

