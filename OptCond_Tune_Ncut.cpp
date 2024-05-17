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

int	const		Ncpu        = 200;				/* total number of CPUs */
int const		Ncut        = 31;				/* cutoff number of Floquet mode */

/* INPUT: physical constants (in the unit of E_c) */

double const	Eta         = 0.001;            /* level broadening width */
double const	Freq        = 0.16;             /* optical frequency */
double const    E0          = 0.001;            /* electric field */
double const    Delta       = 0.5;
double const    Energyk     = 0.5*Delta + 0.2;
double const    Anglek      = 0.5*M_PI;
double const    Omega       = 0.5*Freq;


/****************/
/* MAIN ROUTINE */
/****************/

main(int argc, char **argv)
{
    /* OPEN: saving files */
    
    FILE *f1;
    f1 = fopen("OptCond_Ncut","wt");
    
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
        
        int p, q, s;
        
        gsl_complex z0, z1, z2, z3, z4, z5;
        gsl_complex zz1, zz2, zz3;
        gsl_complex zzz1;
        gsl_complex RetGreen, AdvGreen;
        
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
        gsl_linalg_complex_LU_decomp (InvGr, permGr, &s);
        gsl_linalg_complex_LU_invert (InvGr, permGr, Gr);
        
        /* CALCULATION: matrix inversion to get the advanced Green's function */
        
        gsl_permutation *permGa = gsl_permutation_alloc (Ncut);
        gsl_matrix_complex *Ga = gsl_matrix_complex_alloc (Ncut, Ncut);
        gsl_linalg_complex_LU_decomp (InvGa, permGa, &s);
        gsl_linalg_complex_LU_invert (InvGa, permGa, Ga);
        
        /* SAVE: data on Green's function */
        
        /* LOOP: Floquet mode */
        
        for (p = 0; p <= Ncut-1; p++)
        {
            RetGreen = gsl_matrix_complex_get (Gr, p, p);
            AdvGreen = gsl_matrix_complex_get (Ga, p, p);
            
            fprintf(f1, "%d %e %e %e %e\n", p-(Ncut-1)/2, GSL_REAL(RetGreen), GSL_IMAG(RetGreen), GSL_REAL(AdvGreen), GSL_IMAG(AdvGreen));
        }
        
        fprintf(f1, "\n");
        
        printf("Calculation was done.");
        
        /* FREE: previous allocation for arrays */
        
        gsl_permutation_free (permGr);
        gsl_permutation_free (permGa);
        gsl_matrix_complex_free (InvGr);
        gsl_matrix_complex_free (InvGa);
        gsl_matrix_complex_free (Gr);
        gsl_matrix_complex_free (Ga);
    }
    
    /* CLOSE: saving files */
    
    fclose(f1);
    
    /* FINISH: MPI */
    
    MPI_Finalize();
}
