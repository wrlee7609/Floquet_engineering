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

/* INPUT: numerical constants */

int const       mode        = 1;
int	const		Ncpu        = 200;				/* total number of CPUs */
int const		Ncut        = 31;				/* cutoff number of Floquet mode */

/* INPUT: physical constants (in the unit of E_c) */

double const	Temp        = 0.01;             /* temperature */
double const	Eta         = 0.001;            /* level broadening width */
double const	Freq        = 0.10;             /* optical frequency */
int const       harmonics   = 1;                /* number of Fourier harmonics for optical conductivity */
double const    E0          = 0.001;            /* electric field */
double const    Delta       = 0.2;
double const    Energyk     = 0.5*Delta + 0.3;
double const    Anglek      = 0.5*M_PI;
double const    Omega       = 0.4*Freq;


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
        
        /* DECLARATION: variables and arrays */
        
        int j, m, n, l, s, p, q;
        int sgn;
        double sgn_band;
        
        gsl_complex z0, z1, z2, z3, z4, z5, z6, z7, z8, z9;
        gsl_complex zz1, zz2, zz3, zz4, zz5, zz6, zz7, zz8, zz9, zz10, zz11, zz12, zz13, zz14;
        gsl_complex zz15, zz16, zz17, zz18, zz19, zz20, zz21, zz22, zz23, zz24, zz25, zz26;
        gsl_complex zzz1, zzz2, zzz3;
        gsl_complex gr1, gr2_1, gr2_2, gr2, gr3_1, gr3_2, gr3, gr4_1_1, gr4_1_2, gr4_2_1, gr4_2_2, gr4_1, gr4_2, gr4;
        gsl_complex ga1, ga2_1, ga2_2, ga2;
        gsl_complex Weight_Xp, Weight_Xn, Weight_Yp, Weight_Yn, Invg;
        gsl_complex Weight_X, Weight_Y, LesG, LesGp, LesGn;
        
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
                    
                    GSL_SET_COMPLEX (&z0, Omega + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, Eta);
                    
                    GSL_SET_COMPLEX (&z1, pow(Energyk, 2) - pow(0.5*Delta, 2), 0);
                    GSL_SET_COMPLEX (&z2, pow(0.5 * E0 / Freq, 2), 0);
                    
                    GSL_SET_COMPLEX (&z3, Omega + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                    GSL_SET_COMPLEX (&z4, Omega + (p+1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                    GSL_SET_COMPLEX (&z5, Omega + (p-1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                    
                    zz1 = gsl_complex_div (z1, z3);
                    zz2 = gsl_complex_div (z2, z4);
                    zz3 = gsl_complex_div (z2, z5);
                    
                    zzz1 = gsl_complex_sub (z0, zz1);
                    zzz1 = gsl_complex_sub (zzz1, zz2);
                    zzz1 = gsl_complex_sub (zzz1, zz3);
                    
                } else if (p == q + 1) {
                    
                    GSL_SET_COMPLEX (&z1, 0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek),
                                     - 0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek));
                    GSL_SET_COMPLEX (&z2, - 0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek),
                                     - 0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek));
                    
                    GSL_SET_COMPLEX (&z3, Omega + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                    GSL_SET_COMPLEX (&z4, Omega + (p+1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                    
                    zz1 = gsl_complex_div (z1, z3);
                    zz2 = gsl_complex_div (z2, z4);
                    
                    zzz1 = gsl_complex_add (zz1, zz2);
                    
                } else if (p == q - 1) {
                    
                    GSL_SET_COMPLEX (&z1, - 0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek),
                                     0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek));
                    GSL_SET_COMPLEX (&z2, 0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek),
                                     0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek));
                    
                    GSL_SET_COMPLEX (&z3, Omega + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                    GSL_SET_COMPLEX (&z4, Omega + (p-1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                    
                    zz1 = gsl_complex_div (z1, z3);
                    zz2 = gsl_complex_div (z2, z4);
                    
                    zzz1 = gsl_complex_add (zz1, zz2);
                    
                } else if (p == q + 2) {
                    
                    GSL_SET_COMPLEX (&z1, pow(0.5 * E0 / Freq, 2), 0);
                    GSL_SET_COMPLEX (&z2, Omega + (p+1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                    
                    zzz1 = gsl_complex_div (z1, z2);
                    
                } else if (p == q - 2) {
                    
                    GSL_SET_COMPLEX (&z1, pow(0.5 * E0 / Freq, 2), 0);
                    GSL_SET_COMPLEX (&z2, Omega + (p-1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                    
                    zzz1 = gsl_complex_div (z1, z2);
                    
                } else {
                    
                    GSL_SET_COMPLEX (&zzz1, 0, 0);
                }
                
                gsl_matrix_complex_set (InvGr, p, q, zzz1);
                
                /* INPUT: inverse advanced Green's function */
                
                if (p == q) {
                    
                    GSL_SET_COMPLEX (&z0, Omega + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, - Eta);
                    
                    GSL_SET_COMPLEX (&z1, pow(Energyk, 2) - pow(0.5*Delta, 2), 0);
                    GSL_SET_COMPLEX (&z2, pow(0.5 * E0 / Freq, 2), 0);
                    
                    GSL_SET_COMPLEX (&z3, Omega + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                    GSL_SET_COMPLEX (&z4, Omega + (p+1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                    GSL_SET_COMPLEX (&z5, Omega + (p-1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                    
                    zz1 = gsl_complex_div (z1, z3);
                    zz2 = gsl_complex_div (z2, z4);
                    zz3 = gsl_complex_div (z2, z5);
                    
                    zzz1 = gsl_complex_sub (z0, zz1);
                    zzz1 = gsl_complex_sub (zzz1, zz2);
                    zzz1 = gsl_complex_sub (zzz1, zz3);
                    
                } else if (p == q + 1) {
                    
                    GSL_SET_COMPLEX (&z1, 0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek),
                                     - 0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek));
                    GSL_SET_COMPLEX (&z2, - 0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek),
                                     - 0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek));
                    
                    GSL_SET_COMPLEX (&z3, Omega + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                    GSL_SET_COMPLEX (&z4, Omega + (p+1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                    
                    zz1 = gsl_complex_div (z1, z3);
                    zz2 = gsl_complex_div (z2, z4);
                    
                    zzz1 = gsl_complex_add (zz1, zz2);
                    
                } else if (p == q - 1) {
                    
                    GSL_SET_COMPLEX (&z1, - 0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek),
                                     0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek));
                    GSL_SET_COMPLEX (&z2, 0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek),
                                     0.5 * E0 / Freq * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek));
                    
                    GSL_SET_COMPLEX (&z3, Omega + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                    GSL_SET_COMPLEX (&z4, Omega + (p-1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                    
                    zz1 = gsl_complex_div (z1, z3);
                    zz2 = gsl_complex_div (z2, z4);
                    
                    zzz1 = gsl_complex_add (zz1, zz2);
                    
                } else if (p == q + 2) {
                    
                    GSL_SET_COMPLEX (&z1, pow(0.5 * E0 / Freq, 2), 0);
                    GSL_SET_COMPLEX (&z2, Omega + (p+1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                    
                    zzz1 = gsl_complex_div (z1, z2);
                    
                } else if (p == q - 2) {
                    
                    GSL_SET_COMPLEX (&z1, pow(0.5 * E0 / Freq, 2), 0);
                    GSL_SET_COMPLEX (&z2, Omega + (p-1-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, - Eta);
                    
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
        gsl_linalg_complex_LU_decomp (InvGr, permGr, &sgn);
        gsl_linalg_complex_LU_invert (InvGr, permGr, Gr);
        
        /* CALCULATION: matrix inversion to get the advanced Green's function */
        
        gsl_permutation *permGa = gsl_permutation_alloc (Ncut);
        gsl_matrix_complex *Ga = gsl_matrix_complex_alloc (Ncut, Ncut);
        gsl_linalg_complex_LU_decomp (InvGa, permGa, &sgn);
        gsl_linalg_complex_LU_invert (InvGa, permGa, Ga);
        
        if (mode == 1)
        {
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
                    
                    Occ[p] = Eta / M_PI / (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq - sgn_band * Energyk, 2) + pow(Eta, 2))
                    / (exp((Omega + (p-(Ncut-1.0)/2.0)*Freq) / Temp) + 1.0);
                    
                    /* INPUT: weight function Wa */
                    
                    Weight_Wa[p] = 0.5 * (1.0 + sgn_band * 0.5 * Delta / Energyk - (1.0 - sgn_band * 0.5 * Delta / Energyk) * (pow(Energyk, 2) - pow(0.5 * Delta, 2))
                                          / (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, 2) + pow(Eta, 2)))
                    * (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq - Energyk, 2) + pow(Eta, 2))
                    * (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq + Energyk, 2) + pow(Eta, 2))
                    / ((pow(Omega + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, 2) + pow(Eta, 2))
                       * (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2))
                       - pow(pow(Energyk, 2) - pow(0.5 * Delta, 2), 2));
                    
                    /* INPUT: weight function Wb */
                    
                    Weight_Wb[p] = 0.5 * (1.0 - sgn_band * 0.5 * Delta / Energyk - (1.0 + sgn_band * 0.5 * Delta / Energyk) * (pow(Energyk, 2) - pow(0.5 * Delta, 2))
                                          / (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)))
                    * (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq - Energyk, 2) + pow(Eta, 2))
                    * (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq + Energyk, 2) + pow(Eta, 2))
                    / ((pow(Omega + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, 2) + pow(Eta, 2))
                       * (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2))
                       - pow(pow(Energyk, 2) - pow(0.5 * Delta, 2), 2));
                }
                
                /* LOOP: Floquet mode */
                
                for (n = 1; n <= Ncut-2-harmonics; n++)
                {
                    /* INPUT: absorption part of the lesser Green's function with the weight function X */
                    
                    GSL_SET_COMPLEX (&z1, Weight_Wa[n+harmonics] * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek),
                                     - Weight_Wa[n+harmonics] * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek));
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
                    
                    GSL_SET_COMPLEX (&z1, Weight_Wa[n] * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek),
                                     - Weight_Wa[n] * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek));
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
                        
                        GSL_SET_COMPLEX (&Invg, Omega + (n+harmonics-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                        
                        /* 1st term */
                        
                        GSL_SET_COMPLEX (&z1, (pow(pow(Energyk, 2) - pow(0.5*Delta, 2), 1.5) * Weight_Wa[l] + sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2))
                                               * (pow(Omega + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)) * Weight_Wb[l]) * cos(Anglek),
                                         - (pow(pow(Energyk, 2) - pow(0.5*Delta, 2), 1.5) * Weight_Wa[l] + sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2))
                                            * (pow(Omega + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)) * Weight_Wb[l]) * sin(Anglek));
                        
                        zz1 = gsl_complex_div (z1, Invg);
                        zz2 = gsl_complex_mul (zz1, gr1);
                        zz3 = gsl_complex_mul (zz2, ga1);
                        
                        /* 2nd term */
                        
                        GSL_SET_COMPLEX (&z2, 0, 0.5 * E0 / Freq);
                        GSL_SET_COMPLEX (&z3, cos(2.0*Anglek), - sin(2.0*Anglek));
                        
                        zz4 = gsl_complex_div (z2, Invg);
                        
                        zz5 = gsl_complex_mul (gr1, ga2);
                        zz6 = gsl_complex_mul (z3, gr3);
                        zz7 = gsl_complex_sub (gr2, zz6);
                        zz8 = gsl_complex_mul (zz7, ga1);
                        zz9 = gsl_complex_add (zz5, zz8);
                        zz10 = gsl_complex_mul_real (zz9, (pow(Energyk, 2) - pow(0.5*Delta, 2)) * Weight_Wa[l]);
                        
                        zz11 = gsl_complex_mul (gr2, ga1);
                        zz12 = gsl_complex_mul_real (zz11, (pow(Omega + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)) * Weight_Wb[l]);
                        
                        zz13 = gsl_complex_add (zz10, zz12);
                        zz14 = gsl_complex_mul (zz4, zz13);
                        
                        /* 3rd term */
                        
                        GSL_SET_COMPLEX (&z4, - pow(0.5 * E0 / Freq, 2) * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * Weight_Wa[l], 0);
                        GSL_SET_COMPLEX (&z5, cos(2.0*Anglek), sin(2.0*Anglek));
                        
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
                        
                        GSL_SET_COMPLEX (&Invg, Omega + (n-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                        
                        /* 1st term */
                        
                        GSL_SET_COMPLEX (&z1, (pow(pow(Energyk, 2) - pow(0.5*Delta, 2), 1.5) * Weight_Wa[l] + sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2))
                                               * (pow(Omega + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)) * Weight_Wb[l]) * cos(Anglek),
                                         - (pow(pow(Energyk, 2) - pow(0.5*Delta, 2), 1.5) * Weight_Wa[l] + sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2))
                                            * (pow(Omega + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)) * Weight_Wb[l]) * sin(Anglek));
                        
                        zz1 = gsl_complex_div (z1, Invg);
                        zz2 = gsl_complex_mul (zz1, gr1);
                        zz3 = gsl_complex_mul (zz2, ga1);
                        
                        /* 2nd term */
                        
                        GSL_SET_COMPLEX (&z2, 0, 0.5 * E0 / Freq);
                        GSL_SET_COMPLEX (&z3, cos(2.0*Anglek), - sin(2.0*Anglek));
                        
                        zz4 = gsl_complex_div (z2, Invg);
                        
                        zz5 = gsl_complex_mul (gr1, ga2);
                        zz6 = gsl_complex_mul (z3, gr3);
                        zz7 = gsl_complex_sub (gr2, zz6);
                        zz8 = gsl_complex_mul (zz7, ga1);
                        zz9 = gsl_complex_add (zz5, zz8);
                        zz10 = gsl_complex_mul_real (zz9, (pow(Energyk, 2) - pow(0.5*Delta, 2)) * Weight_Wa[l]);
                        
                        zz11 = gsl_complex_mul (gr2, ga1);
                        zz12 = gsl_complex_mul_real (zz11, (pow(Omega + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)) * Weight_Wb[l]);
                        
                        zz13 = gsl_complex_add (zz10, zz12);
                        zz14 = gsl_complex_mul (zz4, zz13);
                        
                        /* 3rd term */
                        
                        GSL_SET_COMPLEX (&z4, - pow(0.5 * E0 / Freq, 2) * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * Weight_Wa[l], 0);
                        GSL_SET_COMPLEX (&z5, cos(2.0*Anglek), sin(2.0*Anglek));
                        
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
        }
        
        if (mode == 2)
        {
            /* LOOP: emission & absorption processes */
            
            for (j = 0; j <= 1; j++)
            {
                /* INITIALIZATION: lesser Green's function */
                
                GSL_SET_COMPLEX (&LesG, 0, 0);
            
                /* LOOP: band index */
                
                for (m = 0; m <= 1; m++)
                {
                    sgn_band = 2.0 * m - 1.0;
                    
                    /* LOOP: Floquet mode */
                    
                    for (p = 0; p <= Ncut-1; p++)
                    {
                        /* INPUT: electron occupation function in the band basis */
                        
                        Occ[p] = Eta / M_PI / (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq - sgn_band * Energyk, 2) + pow(Eta, 2))
                                / (exp((Omega + (p-(Ncut-1.0)/2.0)*Freq) / Temp) + 1.0);
                        
                        /* INPUT: weight function Wa */
                        
                        Weight_Wa[p] = 0.5 * (1.0 + sgn_band * 0.5 * Delta / Energyk - (1.0 - sgn_band * 0.5 * Delta / Energyk) * (pow(Energyk, 2) - pow(0.5 * Delta, 2))
                                        / (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, 2) + pow(Eta, 2)))
                                        * (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq - Energyk, 2) + pow(Eta, 2))
                                        * (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq + Energyk, 2) + pow(Eta, 2))
                                        / ((pow(Omega + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, 2) + pow(Eta, 2))
                                        * (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2))
                                        - pow(pow(Energyk, 2) - pow(0.5 * Delta, 2), 2));
                        
                        /* INPUT: weight function Wb */
                        
                        Weight_Wb[p] = 0.5 * (1.0 - sgn_band * 0.5 * Delta / Energyk - (1.0 + sgn_band * 0.5 * Delta / Energyk) * (pow(Energyk, 2) - pow(0.5 * Delta, 2))
                                        / (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2)))
                                        * (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq - Energyk, 2) + pow(Eta, 2))
                                        * (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq + Energyk, 2) + pow(Eta, 2))
                                        / ((pow(Omega + (p-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, 2) + pow(Eta, 2))
                                        * (pow(Omega + (p-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, 2) + pow(Eta, 2))
                                        - pow(pow(Energyk, 2) - pow(0.5 * Delta, 2), 2));
                    }
                    
                    /* LOOP: Floquet mode */
                    
                    for (n = 0; n <= Ncut-1-harmonics; n++)
                    {
                        /* INITIALIZATION: weight function X */
                        
                        GSL_SET_COMPLEX (&Weight_X, 0, 0);
                        
                        /* LOOP: Floquet mode */
                        
                        for (l = 0; l <= Ncut-1; l++)
                        {
                            if (l == n+(1-j)*harmonics)
                            {
                                GSL_SET_COMPLEX (&z1, Weight_Wa[n+(1-j)*harmonics] * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek),
                                                 - Weight_Wa[n+(1-j)*harmonics] * sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek));
                                
                            } else if (l == n+(1-j)*harmonics + 1) {
                                
                                GSL_SET_COMPLEX (&z1, 0, Weight_Wa[n+(1-j)*harmonics] * 0.5 * E0 / Freq);
                                
                            } else if (l == n+(1-j)*harmonics - 1) {
                                
                                GSL_SET_COMPLEX (&z1, 0, - Weight_Wa[n+(1-j)*harmonics] * 0.5 * E0 / Freq);
                                
                            } else {
                                
                                GSL_SET_COMPLEX (&z1, 0, 0);
                            }
                            
                            Weight_X = gsl_complex_add (Weight_X, gsl_complex_mul (z1, gsl_matrix_complex_get (Ga, l, n+j*harmonics)));
                        }
                        
                        LesG = gsl_complex_add (LesG, gsl_complex_mul_imag (Weight_X, Occ[n+(1-j)*harmonics]));
                        
                        /* LOOP: Floquet mode */
                        
                        for (l = 0; l <= Ncut-1; l++)
                        {
                            /* INITIALIZATION: weight function Y */
                            
                            GSL_SET_COMPLEX (&Weight_Y, 0, 0);
                            
                            GSL_SET_COMPLEX (&z1, Omega + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, Eta);
                            GSL_SET_COMPLEX (&z2, Omega + (l-(Ncut-1.0)/2.0)*Freq + 0.5*Delta, - Eta);
                            GSL_SET_COMPLEX (&z3, Omega + (n+(1-j)*harmonics-(Ncut-1.0)/2.0)*Freq - 0.5*Delta, Eta);
                            
                            z4 = gsl_complex_mul_real (gsl_complex_div (gsl_complex_mul (z1, z2), z3), Weight_Wb[l]);
                            
                            /* LOOP: Floquet mode */
                            
                            for (s = 0; s <= Ncut-1; s++)
                            {
                                if (s == n+(1-j)*harmonics)
                                {
                                    GSL_SET_COMPLEX (&z5, sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek),
                                                     - sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek));
                                    
                                } else if (s == n+(1-j)*harmonics + 1) {
                                    
                                    GSL_SET_COMPLEX (&z5, 0, 0.5 * E0 / Freq);
                                    
                                } else if (s == n+(1-j)*harmonics - 1) {
                                    
                                    GSL_SET_COMPLEX (&z5, 0, - 0.5 * E0 / Freq);
                                    
                                } else {
                                    
                                    GSL_SET_COMPLEX (&z5, 0, 0);
                                }
                                
                                z6 = gsl_matrix_complex_get (Gr, s, l);
                                z7 = gsl_matrix_complex_get (Ga, l, n+j*harmonics);
                                
                                Weight_Y = gsl_complex_add (Weight_Y, gsl_complex_mul (gsl_complex_mul (gsl_complex_mul (z4, z5), z6), z7));
                            }
                            
                            z4 = gsl_complex_mul_real (gsl_complex_inverse (z3), Weight_Wa[l]);
                            
                            /* LOOP: Floquet mode */
                            
                            for (s = 0; s <= Ncut-1; s++)
                            {
                                if (s == n+(1-j)*harmonics)
                                {
                                    GSL_SET_COMPLEX (&z5, sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek),
                                                     - sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek));
                                    
                                } else if (s == n+(1-j)*harmonics + 1) {
                                    
                                    GSL_SET_COMPLEX (&z5, 0, 0.5 * E0 / Freq);
                                    
                                } else if (s == n+(1-j)*harmonics - 1) {
                                    
                                    GSL_SET_COMPLEX (&z5, 0, - 0.5 * E0 / Freq);
                                    
                                } else {
                                    
                                    GSL_SET_COMPLEX (&z5, 0, 0);
                                }
                                
                                /* LOOP: Floquet mode */
                                
                                for (p = 0; p <= Ncut-1; p++)
                                {
                                    if (p == l)
                                    {
                                        GSL_SET_COMPLEX (&z6, sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek),
                                                         sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek));
                                        
                                    } else if (p == l + 1) {
                                        
                                        GSL_SET_COMPLEX (&z6, 0, - 0.5 * E0 / Freq);
                                        
                                    } else if (p == l - 1) {
                                        
                                        GSL_SET_COMPLEX (&z6, 0, 0.5 * E0 / Freq);
                                        
                                    } else {
                                        
                                        GSL_SET_COMPLEX (&z6, 0, 0);
                                    }
                                    
                                    /* LOOP: Floquet mode */
                                    
                                    for (q = 0; q <= Ncut-1; q++)
                                    {
                                        if (q == l)
                                        {
                                            GSL_SET_COMPLEX (&z7, sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * cos(Anglek),
                                                             - sqrt(pow(Energyk, 2) - pow(0.5*Delta, 2)) * sin(Anglek));
                                            
                                        } else if (q == l + 1) {
                                            
                                            GSL_SET_COMPLEX (&z7, 0, 0.5 * E0 / Freq);
                                            
                                        } else if (q == l - 1) {
                                            
                                            GSL_SET_COMPLEX (&z7, 0, - 0.5 * E0 / Freq);
                                            
                                        } else {
                                            
                                            GSL_SET_COMPLEX (&z7, 0, 0);
                                        }
                                        
                                        z8 = gsl_matrix_complex_get (Gr, s, p);
                                        z9 = gsl_matrix_complex_get (Ga, q, n+j*harmonics);
                                        
                                        Weight_Y = gsl_complex_add (Weight_Y,
                                                                    gsl_complex_mul (gsl_complex_mul (gsl_complex_mul (gsl_complex_mul (gsl_complex_mul (z4, z5), z6), z7), z8), z9));
                                    }
                                }
                            }
                            
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
        }
        
        /* PRINT: integrand of optical conductivity */
        
        printf("Re_OptCond_xx = %f\n", Energyk/M_PI/E0 * (- GSL_IMAG (LesGp) - GSL_IMAG (LesGn)));
        printf("Im_OptCond_xx = %f\n", Energyk/M_PI/E0 * (GSL_REAL (LesGp) - GSL_REAL (LesGn)));
        printf("Re_OptCond_xy = %f\n", Energyk/M_PI/E0 * (GSL_REAL (LesGp) + GSL_REAL (LesGn)));
        printf("Im_OptCond_xy = %f\n", Energyk/M_PI/E0 * (GSL_IMAG (LesGp) - GSL_IMAG (LesGn)));
        
        /* FREE: previous allocation for arrays */
        
        gsl_permutation_free (permGr);
        gsl_permutation_free (permGa);
        gsl_matrix_complex_free (InvGr);
        gsl_matrix_complex_free (InvGa);
        gsl_matrix_complex_free (Gr);
        gsl_matrix_complex_free (Ga);
    }
    
    /* FINISH: MPI */
    
    MPI_Finalize();
}
