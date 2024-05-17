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

#include <gsl/gsl_integration.h>

#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>

/* INPUT: numerical constants */

int	const		Ncpu        = 200;				/* total number of CPUs */
int const		Ncut        = 31;				/* cutoff number of Floquet mode */
int const       MCC         = 500000;           /* number of function calls for Monte Carlo integration */

/* INPUT: physical constants (in the unit of E_c) */

double const	Temp        = 0.01;             /* temperature */
double const	Eta         = 0.001;            /* level broadening width */
double const	Freq        = 0.16;             /* optical frequency */
int const       harmonics   = 1;                /* number of Fourier harmonics for optical conductivity */
double const    E0          = 0.001;            /* electric field */
double const    Delta       = 0.5;              /* band gap */


/*************************************************/
/* SUBLOUTINE: integrand of optical conductivity */
/*************************************************/

struct gsl_params {
    
    int comp;
};

double g (double *vars, size_t dim, void *params) {
    
    /* DECLARATION: variables for function g */
     
    /* vars[0]: variable for energy dispersion */
    /* vars[1]: variable for angle in momentum space */
    /* vars[2]: variable for frequency domain */
    
    /* DECLARATION: parameters for function g */
    
    int comp = ((struct gsl_params *) params) -> comp;
    
    /* DECLARATION: variables and arrays */
    
    int p, q, m, n, l, s;
    double sgn_band, OptCond_Int;
    
    gsl_complex z0, z1, z2, z3, z4, z5, z6, z7;
    gsl_complex zz1, zz2, zz3, zz4, zz5, zz6, zz7, zz8, zz9, zz10, zz11, zz12, zz13, zz14;
    gsl_complex zz15, zz16, zz17, zz18, zz19, zz20, zz21, zz22, zz23, zz24, zz25, zz26;
    gsl_complex zzz1, zzz2, zzz3;
    gsl_complex gr1, gr2_1, gr2_2, gr2, gr3_1, gr3_2, gr3, gr4_1_1, gr4_1_2, gr4_2_1, gr4_2_2, gr4_1, gr4_2, gr4;
    gsl_complex ga1, ga2_1, ga2_2, ga2;
    gsl_complex Weight_Xp, Weight_Xn, Weight_Yp, Weight_Yn, LesGp, LesGn, Invg;
    
    double Occ[Ncut], Weight_Wa[Ncut], Weight_Wb[Ncut];
    
    gsl_matrix_complex *InvGr = gsl_matrix_complex_alloc (Ncut, Ncut);
    gsl_matrix_complex *InvGa = gsl_matrix_complex_alloc (Ncut, Ncut);
    
    /* LOOP: Floquet mode */
    
    for (p = 0; p <= Ncut-1; p++)
    {
        /* LOOP: Floquet mode */
        
        for (q = 0; q <= Ncut-1; q++)
        {
            /* INPUT: inverse retarded Green's function */
            
            if (p == q) {
                
                GSL_SET_COMPLEX (&z0, vars[2] + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, Eta);
                
                GSL_SET_COMPLEX (&z1, pow(vars[0], 2) - pow(0.5*Delta, 2), 0);
                GSL_SET_COMPLEX (&z2, pow(0.5 * E0 / Freq, 2), 0);
                
                GSL_SET_COMPLEX (&z3, vars[2] + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                GSL_SET_COMPLEX (&z4, vars[2] + (p+1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                GSL_SET_COMPLEX (&z5, vars[2] + (p-1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                
                zz1 = gsl_complex_div (z1, z3);
                zz2 = gsl_complex_div (z2, z4);
                zz3 = gsl_complex_div (z2, z5);
                
                zzz1 = gsl_complex_sub (z0, zz1);
                zzz1 = gsl_complex_sub (zzz1, zz2);
                zzz1 = gsl_complex_sub (zzz1, zz3);
                
            } else if (p == q + 1) {
                
                GSL_SET_COMPLEX (&z1, 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * sin(vars[1]),
                                 - 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * cos(vars[1]));
                GSL_SET_COMPLEX (&z2, - 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * sin(vars[1]),
                                 - 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * cos(vars[1]));
                
                GSL_SET_COMPLEX (&z3, vars[2] + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                GSL_SET_COMPLEX (&z4, vars[2] + (p+1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                
                zz1 = gsl_complex_div (z1, z3);
                zz2 = gsl_complex_div (z2, z4);
                
                zzz1 = gsl_complex_add (zz1, zz2);
                
            } else if (p == q - 1) {
                
                GSL_SET_COMPLEX (&z1, - 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * sin(vars[1]),
                                 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * cos(vars[1]));
                GSL_SET_COMPLEX (&z2, 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * sin(vars[1]),
                                 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * cos(vars[1]));
                
                GSL_SET_COMPLEX (&z3, vars[2] + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                GSL_SET_COMPLEX (&z4, vars[2] + (p-1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                
                zz1 = gsl_complex_div (z1, z3);
                zz2 = gsl_complex_div (z2, z4);
                
                zzz1 = gsl_complex_add (zz1, zz2);
                
            } else if (p == q + 2) {
                
                GSL_SET_COMPLEX (&z1, pow(0.5 * E0 / Freq, 2), 0);
                GSL_SET_COMPLEX (&z2, vars[2] + (p+1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                
                zzz1 = gsl_complex_div (z1, z2);
                
            } else if (p == q - 2) {
                
                GSL_SET_COMPLEX (&z1, pow(0.5 * E0 / Freq, 2), 0);
                GSL_SET_COMPLEX (&z2, vars[2] + (p-1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                
                zzz1 = gsl_complex_div (z1, z2);
                
            } else {
                
                GSL_SET_COMPLEX (&zzz1, 0, 0);
            }
            
            gsl_matrix_complex_set (InvGr, p, q, zzz1);
            
            /* INPUT: inverse advanced Green's function */
            
            if (p == q) {
                
                GSL_SET_COMPLEX (&z0, vars[2] + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, - Eta);
                
                GSL_SET_COMPLEX (&z1, pow(vars[0], 2) - pow(0.5*Delta, 2), 0);
                GSL_SET_COMPLEX (&z2, pow(0.5 * E0 / Freq, 2), 0);
                
                GSL_SET_COMPLEX (&z3, vars[2] + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                GSL_SET_COMPLEX (&z4, vars[2] + (p+1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                GSL_SET_COMPLEX (&z5, vars[2] + (p-1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                
                zz1 = gsl_complex_div (z1, z3);
                zz2 = gsl_complex_div (z2, z4);
                zz3 = gsl_complex_div (z2, z5);
                
                zzz1 = gsl_complex_sub (z0, zz1);
                zzz1 = gsl_complex_sub (zzz1, zz2);
                zzz1 = gsl_complex_sub (zzz1, zz3);
                
            } else if (p == q + 1) {
                
                GSL_SET_COMPLEX (&z1, 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * sin(vars[1]),
                                 - 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * cos(vars[1]));
                GSL_SET_COMPLEX (&z2, - 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * sin(vars[1]),
                                 - 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * cos(vars[1]));
                
                GSL_SET_COMPLEX (&z3, vars[2] + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                GSL_SET_COMPLEX (&z4, vars[2] + (p+1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                
                zz1 = gsl_complex_div (z1, z3);
                zz2 = gsl_complex_div (z2, z4);
                
                zzz1 = gsl_complex_add (zz1, zz2);
                
            } else if (p == q - 1) {
                
                GSL_SET_COMPLEX (&z1, - 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * sin(vars[1]),
                                 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * cos(vars[1]));
                GSL_SET_COMPLEX (&z2, 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * sin(vars[1]),
                                 0.5 * E0 / Freq * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * cos(vars[1]));
                
                GSL_SET_COMPLEX (&z3, vars[2] + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                GSL_SET_COMPLEX (&z4, vars[2] + (p-1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                
                zz1 = gsl_complex_div (z1, z3);
                zz2 = gsl_complex_div (z2, z4);
                
                zzz1 = gsl_complex_add (zz1, zz2);
                
            } else if (p == q + 2) {
                
                GSL_SET_COMPLEX (&z1, pow(0.5 * E0 / Freq, 2), 0);
                GSL_SET_COMPLEX (&z2, vars[2] + (p+1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                
                zzz1 = gsl_complex_div (z1, z2);
                
            } else if (p == q - 2) {
                
                GSL_SET_COMPLEX (&z1, pow(0.5 * E0 / Freq, 2), 0);
                GSL_SET_COMPLEX (&z2, vars[2] + (p-1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                
                zzz1 = gsl_complex_div (z1, z2);
                
            } else {
                
                GSL_SET_COMPLEX (&zzz1, 0, 0);
            }
            
            gsl_matrix_complex_set (InvGa, p, q, zzz1);
        }
    }
    
    /* CALCULATION: matrix inversion to get the retarded Green's function */
    
    gsl_permutation *permGr = gsl_permutation_alloc (Ncut);
    gsl_matrix_complex *Gr = gsl_matrix_complex_alloc (Ncut, Ncut);
    gsl_linalg_complex_LU_decomp (InvGr, permGr, &s);
    gsl_linalg_complex_LU_invert (InvGr, permGr, Gr);
    
    /* CALCULATION: matrix inversion to get the advanced Green's function */
    
    gsl_permutation *permGa = gsl_permutation_alloc (Ncut);
    gsl_matrix_complex *Ga = gsl_matrix_complex_alloc (Ncut, Ncut);
    gsl_linalg_complex_LU_decomp (InvGa, permGa, &s);
    gsl_linalg_complex_LU_invert (InvGa, permGa, Ga);
    
    /* INITIALIZATION: lesser Green's function */
    
    GSL_SET_COMPLEX (&LesGp, 0, 0);
    GSL_SET_COMPLEX (&LesGn, 0, 0);
    
    /* LOOP: band index */
    
    for (m = 0; m <= 1; m++)
    {
        sgn_band = 2.0 * m - 1.0;
        
        /* LOOP: Floquet mode */
        
        for (p = 0; p <= Ncut-1; p++)
        {
            /* INPUT: electron occupation function in the band basis */
            
            Occ[p] = Eta / M_PI / (pow(vars[2] + (p-(Ncut-1.0)/2.0)*Freq - sgn_band * vars[0], 2) + pow(Eta, 2))
                    / (exp((vars[2] + (p-(Ncut-1.0)/2.0)*Freq) / Temp) + 1.0);
            
            /* INPUT: weight function Wa */
            
            Weight_Wa[p] = 0.5 * (1.0 + sgn_band * 0.5 * Delta / vars[0] - (1.0 - sgn_band * 0.5 * Delta / vars[0]) * (pow(vars[0], 2) - pow(0.5 * Delta, 2))
                            / (pow(vars[2] + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, 2) + pow(Eta, 2)))
                            * (pow(vars[2] + (p-(Ncut-1.0)/2.0)*Freq - vars[0], 2) + pow(Eta, 2))
                            * (pow(vars[2] + (p-(Ncut-1.0)/2.0)*Freq + vars[0], 2) + pow(Eta, 2))
                            / ((pow(vars[2] + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, 2) + pow(Eta, 2))
                            * (pow(vars[2] + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2))
                            - pow(pow(vars[0], 2) - pow(0.5 * Delta, 2), 2));
            
            /* INPUT: weight function Wb */
            
            Weight_Wb[p] = 0.5 * (1.0 - sgn_band * 0.5 * Delta / vars[0] - (1.0 + sgn_band * 0.5 * Delta / vars[0]) * (pow(vars[0], 2) - pow(0.5 * Delta, 2))
                            / (pow(vars[2] + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)))
                            * (pow(vars[2] + (p-(Ncut-1.0)/2.0)*Freq - vars[0], 2) + pow(Eta, 2))
                            * (pow(vars[2] + (p-(Ncut-1.0)/2.0)*Freq + vars[0], 2) + pow(Eta, 2))
                            / ((pow(vars[2] + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, 2) + pow(Eta, 2))
                            * (pow(vars[2] + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2))
                            - pow(pow(vars[0], 2) - pow(0.5 * Delta, 2), 2));
        }
        
        /* LOOP: Floquet mode */
        
        for (n = 1; n <= Ncut-2-harmonics; n++)
        {
            /* INPUT: absorption part of the lesser Green's function with the weight function X */
            
            GSL_SET_COMPLEX (&z1, Weight_Wa[n+harmonics] * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * cos(vars[1]),
                             - Weight_Wa[n+harmonics] * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * sin(vars[1]));
            z2 = gsl_matrix_complex_get (Ga, n+harmonics, n);
            zz1 = gsl_complex_mul (z1, z2);
            
            GSL_SET_COMPLEX (&z3, 0, Weight_Wa[n+harmonics] * 0.5 * E0 / Freq);
            z4 = gsl_matrix_complex_get (Ga, n+harmonics+1, n);
            z5 = gsl_matrix_complex_get (Ga, n+harmonics-1, n);
            z6 = gsl_complex_sub (z4, z5);
            zz2 = gsl_complex_mul (z3, z6);
            
            Weight_Xp = gsl_complex_add (zz1, zz2);
            
            z7 = gsl_complex_mul_imag (Weight_Xp, Occ[n+harmonics]);
            LesGp = gsl_complex_add (LesGp, z7);
            
            /* INPUT: emission part of the lesser Green's function with the weight function X */
            
            GSL_SET_COMPLEX (&z1, Weight_Wa[n] * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * cos(vars[1]),
                             - Weight_Wa[n] * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * sin(vars[1]));
            z2 = gsl_matrix_complex_get (Ga, n, n+harmonics);
            zz1 = gsl_complex_mul (z1, z2);
            
            GSL_SET_COMPLEX (&z3, 0, Weight_Wa[n] * 0.5 * E0 / Freq);
            z4 = gsl_matrix_complex_get (Ga, n+1, n+harmonics);
            z5 = gsl_matrix_complex_get (Ga, n-1, n+harmonics);
            z6 = gsl_complex_sub (z4, z5);
            zz2 = gsl_complex_mul (z3, z6);
            
            Weight_Xn = gsl_complex_add (zz1, zz2);
            
            z7 = gsl_complex_mul_imag (Weight_Xn, Occ[n]);
            LesGn = gsl_complex_add (LesGn, z7);
            
            /* LOOP: Floquet mode */
            
            for (l = 1; l <= Ncut-2; l++)
            {
                /* INPUT: absorption part of the lesser Green's function with the weight function Y */
                
                /* common factors */
                
                gr1 = gsl_matrix_complex_get (Gr, n+harmonics, l);
                gr2_1 = gsl_matrix_complex_get (Gr, n+harmonics+1, l);
                gr2_2 = gsl_matrix_complex_get (Gr, n+harmonics-1, l);
                gr2 = gsl_complex_sub (gr2_1, gr2_2);
                gr3_1 = gsl_matrix_complex_get (Gr, n+harmonics, l+1);
                gr3_2 = gsl_matrix_complex_get (Gr, n+harmonics, l-1);
                gr3 = gsl_complex_sub (gr3_1, gr3_2);
                gr4_1_1 = gsl_matrix_complex_get (Gr, n+harmonics+1, l+1);
                gr4_1_2 = gsl_matrix_complex_get (Gr, n+harmonics+1, l-1);
                gr4_2_1 = gsl_matrix_complex_get (Gr, n+harmonics-1, l+1);
                gr4_2_2 = gsl_matrix_complex_get (Gr, n+harmonics-1, l-1);
                gr4_1 = gsl_complex_sub (gr4_1_1, gr4_1_2);
                gr4_2 = gsl_complex_sub (gr4_2_1, gr4_2_2);
                gr4 = gsl_complex_sub (gr4_1, gr4_2);
                
                ga1 = gsl_matrix_complex_get (Ga, l, n);
                ga2_1 = gsl_matrix_complex_get (Ga, l+1, n);
                ga2_2 = gsl_matrix_complex_get (Ga, l-1, n);
                ga2 = gsl_complex_sub (ga2_1, ga2_2);
                
                GSL_SET_COMPLEX (&Invg, vars[2] + (n+harmonics-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                
                /* 1st term */
                
                GSL_SET_COMPLEX (&z1, (pow(pow(vars[0], 2) - pow(0.5*Delta, 2), 1.5) * Weight_Wa[l] + sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2))
                                    * (pow(vars[2] + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)) * Weight_Wb[l]) * cos(vars[1]),
                                    - (pow(pow(vars[0], 2) - pow(0.5*Delta, 2), 1.5) * Weight_Wa[l] + sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2))
                                    * (pow(vars[2] + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)) * Weight_Wb[l]) * sin(vars[1]));
                
                zz1 = gsl_complex_div (z1, Invg);
                zz2 = gsl_complex_mul (zz1, gr1);
                zz3 = gsl_complex_mul (zz2, ga1);
                
                /* 2nd term */
                
                GSL_SET_COMPLEX (&z2, 0, 0.5 * E0 / Freq);
                GSL_SET_COMPLEX (&z3, cos(2.0*vars[1]), - sin(2.0*vars[1]));
                
                zz4 = gsl_complex_div (z2, Invg);
                
                zz5 = gsl_complex_mul (gr1, ga2);
                zz6 = gsl_complex_mul (z3, gr3);
                zz7 = gsl_complex_sub (gr2, zz6);
                zz8 = gsl_complex_mul (zz7, ga1);
                zz9 = gsl_complex_add (zz5, zz8);
                zz10 = gsl_complex_mul_real (zz8, (pow(vars[0], 2) - pow(0.5*Delta, 2)) * Weight_Wa[l]);
                
                zz11 = gsl_complex_mul (gr2, ga1);
                zz12 = gsl_complex_mul_real (zz11, (pow(vars[2] + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)) * Weight_Wb[l]);
                
                zz13 = gsl_complex_add (zz10, zz12);
                zz14 = gsl_complex_mul (zz4, zz13);
                
                /* 3rd term */
                
                GSL_SET_COMPLEX (&z4, - pow(0.5 * E0 / Freq, 2) * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * Weight_Wa[l], 0);
                GSL_SET_COMPLEX (&z5, cos(2.0*vars[1]), sin(2.0*vars[1]));
                
                zz15 = gsl_complex_div (z4, Invg);
                
                zz16 = gsl_complex_mul (z5, gr2);
                zz17 = gsl_complex_mul (z3, gr3);
                zz18 = gsl_complex_sub (zz16, zz17);
                zz19 = gsl_complex_mul (zz18, ga2);
                
                zz20 = gsl_complex_mul (z3, gr4);
                zz21 = gsl_complex_mul (zz20, ga1);
                
                zz22 = gsl_complex_sub (zz19, zz21);
                zz23 = gsl_complex_mul (zz15, zz22);
                
                /* 4th term */
                
                GSL_SET_COMPLEX (&z6, 0, pow(0.5 * E0 / Freq, 3) * Weight_Wa[l]);
                
                zz24 = gsl_complex_div (z6, Invg);
                zz25 = gsl_complex_mul (zz24, gr4);
                zz26 = gsl_complex_mul (zz25, ga2);
                
                /* sum up */
                
                zzz1 = gsl_complex_add (zz3, zz14);
                zzz2 = gsl_complex_add (zz23, zz26);
                Weight_Yp = gsl_complex_add (zzz1, zzz2);
                zzz3 = gsl_complex_mul_imag (Weight_Yp, Occ[l]);
                LesGp = gsl_complex_add (LesGp, zzz3);
                
                /* INPUT: emission part of the lesser Green's function with the weight function Y */
                
                /* common factors */
                
                gr1 = gsl_matrix_complex_get (Gr, n, l);
                gr2_1 = gsl_matrix_complex_get (Gr, n+1, l);
                gr2_2 = gsl_matrix_complex_get (Gr, n-1, l);
                gr2 = gsl_complex_sub (gr2_1, gr2_2);
                gr3_1 = gsl_matrix_complex_get (Gr, n, l+1);
                gr3_2 = gsl_matrix_complex_get (Gr, n, l-1);
                gr3 = gsl_complex_sub (gr3_1, gr3_2);
                gr4_1_1 = gsl_matrix_complex_get (Gr, n+1, l+1);
                gr4_1_2 = gsl_matrix_complex_get (Gr, n+1, l-1);
                gr4_2_1 = gsl_matrix_complex_get (Gr, n-1, l+1);
                gr4_2_2 = gsl_matrix_complex_get (Gr, n-1, l-1);
                gr4_1 = gsl_complex_sub (gr4_1_1, gr4_1_2);
                gr4_2 = gsl_complex_sub (gr4_2_1, gr4_2_2);
                gr4 = gsl_complex_sub (gr4_1, gr4_2);
                
                ga1 = gsl_matrix_complex_get (Ga, l, n+harmonics);
                ga2_1 = gsl_matrix_complex_get (Ga, l+1, n+harmonics);
                ga2_2 = gsl_matrix_complex_get (Ga, l-1, n+harmonics);
                ga2 = gsl_complex_sub (ga2_1, ga2_2);
                
                GSL_SET_COMPLEX (&Invg, vars[2] + (n-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                
                /* 1st term */
                
                GSL_SET_COMPLEX (&z1, (pow(pow(vars[0], 2) - pow(0.5*Delta, 2), 1.5) * Weight_Wa[l] + sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2))
                                    * (pow(vars[2] + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)) * Weight_Wb[l]) * cos(vars[1]),
                                    - (pow(pow(vars[0], 2) - pow(0.5*Delta, 2), 1.5) * Weight_Wa[l] + sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2))
                                    * (pow(vars[2] + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)) * Weight_Wb[l]) * sin(vars[1]));
                
                zz1 = gsl_complex_div (z1, Invg);
                zz2 = gsl_complex_mul (zz1, gr1);
                zz3 = gsl_complex_mul (zz2, ga1);
                
                /* 2nd term */
                
                GSL_SET_COMPLEX (&z2, 0, 0.5 * E0 / Freq);
                GSL_SET_COMPLEX (&z3, cos(2.0*vars[1]), - sin(2.0*vars[1]));
                
                zz4 = gsl_complex_div (z2, Invg);
                
                zz5 = gsl_complex_mul (gr1, ga2);
                zz6 = gsl_complex_mul (z3, gr3);
                zz7 = gsl_complex_sub (gr2, zz6);
                zz8 = gsl_complex_mul (zz7, ga1);
                zz9 = gsl_complex_add (zz5, zz8);
                zz10 = gsl_complex_mul_real (zz8, (pow(vars[0], 2) - pow(0.5*Delta, 2)) * Weight_Wa[l]);
                
                zz11 = gsl_complex_mul (gr2, ga1);
                zz12 = gsl_complex_mul_real (zz11, (pow(vars[2] + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)) * Weight_Wb[l]);
                
                zz13 = gsl_complex_add (zz10, zz12);
                zz14 = gsl_complex_mul (zz4, zz13);
                
                /* 3rd term */
                
                GSL_SET_COMPLEX (&z4, - pow(0.5 * E0 / Freq, 2) * sqrt(pow(vars[0], 2) - pow(0.5*Delta, 2)) * Weight_Wa[l], 0);
                GSL_SET_COMPLEX (&z5, cos(2.0*vars[1]), sin(2.0*vars[1]));
                
                zz15 = gsl_complex_div (z4, Invg);
                
                zz16 = gsl_complex_mul (z5, gr2);
                zz17 = gsl_complex_mul (z3, gr3);
                zz18 = gsl_complex_sub (zz16, zz17);
                zz19 = gsl_complex_mul (zz18, ga2);
                
                zz20 = gsl_complex_mul (z3, gr4);
                zz21 = gsl_complex_mul (zz20, ga1);
                
                zz22 = gsl_complex_sub (zz19, zz21);
                zz23 = gsl_complex_mul (zz15, zz22);
                
                /* 4th term */
                
                GSL_SET_COMPLEX (&z6, 0, pow(0.5 * E0 / Freq, 3) * Weight_Wa[l]);
                
                zz24 = gsl_complex_div (z6, Invg);
                zz25 = gsl_complex_mul (zz24, gr4);
                zz26 = gsl_complex_mul (zz25, ga2);
                
                /* sum up */
                
                zzz1 = gsl_complex_add (zz3, zz14);
                zzz2 = gsl_complex_add (zz23, zz26);
                Weight_Yn = gsl_complex_add (zzz1, zzz2);
                zzz3 = gsl_complex_mul_imag (Weight_Yn, Occ[l]);
                LesGn = gsl_complex_add (LesGn, zzz3);
            }
        }
    }
    
    /* INPUT: integrand of optical conductivity */
    
    if (comp == 0) {    /* Re_OptCond_xx */
    
        OptCond_Int = vars[0]/M_PI/E0 * (- GSL_IMAG (LesGp) - GSL_IMAG (LesGn));
    }
    
    if (comp == 1) {    /* Im_OptCond_xx */
        
        OptCond_Int = vars[0]/M_PI/E0 * (GSL_REAL (LesGp) - GSL_REAL (LesGn));
    }
    
    if (comp == 2) {    /* Re_OptCond_xy */
        
        OptCond_Int = vars[0]/M_PI/E0 * (GSL_REAL (LesGp) + GSL_REAL (LesGn));
    }
    
    if (comp == 3) {    /* Im_OptCond_xy */
        
        OptCond_Int = vars[0]/M_PI/E0 * (GSL_IMAG (LesGp) - GSL_IMAG (LesGn));
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


/************************************************/
/* SUBLOUTINE: optical conductivity vs band gap */
/************************************************/

double Subroutine_OptCond (int rank)
{
    /* DECLARATION: variables and arrays */
    
    int comp;
    double result, error;
    
    double xl[3] = {0.5*Delta, 0, -0.5*Freq};
    double xu[3] = {1.0, 2.0*M_PI, 0.5*Freq};
    
    /* LOOP: component of optical conductivity */
    
    for (comp = 0; comp <= 3; comp++)
    {
        /* INTEGRATION: Monte Carlo Method (PLAIN / MISER / VEGAS) */
        
        struct gsl_params params_p = { comp };
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
        
        printf ("MCC = %d, comp = %d, result = %e, error = %e\n", MCC, comp, result, error);
    }
    
    return result;
}


/****************/
/* MAIN ROUTINE */
/****************/

main(int argc, char **argv)
{
    /* INITIALIZATION: MPI */
    
    int rank;		/* Rank of my CPU */
    int source;		/* CPU sending message */
    int dest = 0;	/* CPU receiving message */
    int tag = 0;	/* Indicator distinguishing messages */
    
    MPI_Status status;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    /* PARALLELIZING: MPI */
    
    if (rank == 0) {
        
        Subroutine_OptCond (rank);
    }
    
    /* FINISH: MPI */
    
    MPI_Finalize();
}
