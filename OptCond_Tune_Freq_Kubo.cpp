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

/* INPUT: numerical constants */

int	const		Ncpu        = 20;				/* total number of CPUs 200 */
int const       Freqcut     = Ncpu;             /* grid number of optical frequency */
double const    dFreq       = 0.03;             /* grid size of optical frequency */
double const    Freq0       = 0.4;              /* initial value inside the window for optical frequency */

/* INPUT: physical constants (in the unit of Delta) */

double const	Temp        = 0.01;             /* temperature */
double const	Eta         = 0.004;            /* level broadening width */
double const    Ec          = 10;               /* Dirac band cutoff */
double const    Field       = 0.0001;           /* electric field */


/*************************************************/
/* SUBLOUTINE: integrand of optical conductivity */
/*************************************************/

struct gsl_params {
    
    int comp, iFreq;
};

double g_Kubo (double x, void * params) {
    
    /* DECLARATION: parameters for function g_Kubo */
    
    int comp = ((struct gsl_params *) params) -> comp;
    int iFreq = ((struct gsl_params *) params) -> iFreq;
    
    double Freq = Freq0 + iFreq * dFreq;
    
    /* DECLARATION: variables */
    
    double Prop_P_Re = (2.0 * x + Freq) / (pow(2.0 * x + Freq, 2) + pow(Eta, 2));
    double Prop_N_Re = (2.0 * x - Freq) / (pow(2.0 * x - Freq, 2) + pow(Eta, 2));
    double Prop_0_Re = Freq / (pow(Freq, 2) + pow(Eta, 2));
    
    double Prop_P_Im = Eta / M_PI / (pow(2.0 * x + Freq, 2) + pow(Eta, 2));
    double Prop_N_Im = Eta / M_PI / (pow(2.0 * x - Freq, 2) + pow(Eta, 2));
    double Prop_0_Im = Eta / M_PI / (pow(Freq, 2) + pow(Eta, 2));
    
    double Fermi_Diff = 1.0 / (exp(- x / Temp) + 1.0) - 1.0 / (exp(x / Temp) + 1.0);
    double Fermi_Grad = exp(- x / Temp) / Temp / pow(exp(- x / Temp) + 1.0, 2) + exp(x / Temp) / Temp / pow(exp(x / Temp) + 1.0, 2);
    
    double OptCond_Int;
    
    /* INPUT: integrand of optical conductivity */
    
    if (comp == 0) {    /* Re_OptCond_xx */
        
        OptCond_Int = 0.25 * M_PI * (1.0 + pow(0.5/x, 2)) * (Prop_P_Im + Prop_N_Im) * Fermi_Diff
                    + 0.5 * M_PI * Prop_0_Im * x * (1.0 - pow(0.5/x, 2)) * Fermi_Grad;
    }
    
    if (comp == 1) {    /* Im_OptCond_xx */
        
        OptCond_Int = 0.25 * (1.0 + pow(0.5/x, 2)) * (Prop_P_Re - Prop_N_Re) * Fermi_Diff
                    + 0.5 * Prop_0_Re * x * (1.0 - pow(0.5/x, 2)) * Fermi_Grad;
    }
    
    if (comp == 2) {    /* Re_OptCond_xy */
        
        OptCond_Int = - 0.25 / x * (Prop_P_Re + Prop_N_Re) * Fermi_Diff;
    }
    
    if (comp == 3) {    /* Im_OptCond_xy */
        
        OptCond_Int = 0.25 * M_PI / x * (Prop_P_Im - Prop_N_Im) * Fermi_Diff;
    }
    
    return OptCond_Int;
}


/************************************************/
/* SUBLOUTINE: optical conductivity vs band gap */
/************************************************/

double *Subroutine_OptCond_Kubo (int rank)
{
    /* DECLARATION: variables and arrays */
    
    int comp, iFreq;
    double Freq, result, error;
    static double OptCond_Sub[4*Freqcut/Ncpu];
    
    /* LOOP: electric field */
    
    for (iFreq = rank*Freqcut/Ncpu; iFreq <= (rank+1)*Freqcut/Ncpu-1; iFreq++)
    {
        Freq = Freq0 + iFreq * dFreq;
        
        /* LOOP: component of optical conductivity */
        
        for (comp = 0; comp <= 3; comp++)
        {
            /* INTEGRATION: QAG adaptive integration */
            
            struct gsl_params params_p = { comp, iFreq };
            
            gsl_function G;
            G.function = &g_Kubo;
            G.params = &params_p;
            
            gsl_integration_workspace *w = gsl_integration_workspace_alloc (1000);
            gsl_integration_qag (&G, 0.5, Ec, 0, 0.000001, 1000, 3, w, &result, &error);
            gsl_integration_workspace_free (w);
            
            OptCond_Sub[iFreq - rank * Freqcut/Ncpu + comp * Freqcut/Ncpu] = result;
            
            printf ("optical frequency = %f, result = %e, error = %e\n", Freq, result, error);
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
    f1 = fopen("OptCond_Ec10_T001_E00001_Kubo","wt");
    
    /* INITIALIZATION: MPI */
    
    int rank;		/* Rank of my CPU */
    int source;		/* CPU sending message */
    int dest = 0;	/* CPU receiving message */
    int tag = 0;	/* Indicator distinguishing messages */
    
    MPI_Status status;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    /* DECLARATION: variables and arrays */
    
    int comp, iFreq;
    double Freq;
    double *OptCond_Sub;
    double OptCond[4*Freqcut];
    
    /* PARALLELIZING: MPI */
    
    if (rank >= 1 && rank <= Ncpu-1) {
        
        OptCond_Sub = Subroutine_OptCond_Kubo (rank);
        MPI_Send (OptCond_Sub, 4*Freqcut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
        
    } else if (rank == 0) {
        
        for (source = 0; source <= Ncpu-1; source++)
        {
            if (source == 0) {
                
                OptCond_Sub = Subroutine_OptCond_Kubo (source);
                
            } else {
                
                MPI_Recv (OptCond_Sub, 4*Freqcut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
            }
            
            /* LOOP: energy gap */
            
            for (iFreq = source*Freqcut/Ncpu; iFreq <= (source+1)*Freqcut/Ncpu-1; iFreq++)
            {
                /* LOOP: component of optical conductivity */
                
                for (comp = 0; comp <= 3; comp++)
                {
                    OptCond[iFreq + comp * Freqcut] = OptCond_Sub[iFreq - source * Freqcut/Ncpu + comp * Freqcut/Ncpu];
                }
            }
        }
    }
    
    /* SAVE: data on optical conductivity */
    
    if (rank == 0) {
        
        /* LOOP: optical frequency */
        
        for (iFreq = 0; iFreq <= Freqcut-1; iFreq++)
        {
            Freq = Freq0 + iFreq * dFreq;
            
            fprintf(f1, "%f %e %e %e %e\n", Freq,
                    OptCond[iFreq + 0 * Freqcut],
                    OptCond[iFreq + 1 * Freqcut],
                    OptCond[iFreq + 2 * Freqcut],
                    OptCond[iFreq + 3 * Freqcut]);
        }
        
        fprintf(f1, "\n");
        
        printf("Calculation was done.");
    }
    
    /* CLOSE: saving files */
    
    fclose(f1);
    
    /* FINISH: MPI */
    
    MPI_Finalize();
}

