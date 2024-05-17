/************************************************************************************************/
/************* [System] 2D Bloch oscillator (HgTe/CdTe quantum wells)			  ***************/
/************* [Method] Floquet-Keldysh Green's function						  ***************/
/************* [Programmer] Dr. Woo-Ram Lee (wrlee@kias.re.kr)					  ***************/
/************************************************************************************************/

// MPI routines for parallel computing

#include <mpi.h>

// Standard libraries

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// GSL libraries

#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>

// Define complex-valued vector

#define RE1(z,i)((z)[2*(i)])
#define IM1(z,i)((z)[2*(i)+1])

// Numerical constants

double const	ScaFac  = 1.0;				// scaling factor for the side direction

int	const		Ncpu    = 15;				// total number of CPUs

int const		Iband   = Ncpu * 45;		// grid number of frequency domain in a bandwidth
int const		Ncut    = 75;				// grid number of Floquet mode, 261
int const		Fmax    = Ncpu * 7;			// Field/Domega
int const		Icut    = Ncut * Fmax;		// grid number of frequency domain in a full range
double const	Domega  = ScaFac/Iband;		// grid in frequency domain

int const		Kcut    = Ncpu * 7;			// grid number of k_perp

int const		K1band	= Ncpu * 55;		// grid number of k_para within 1 B.W.
int const		K1cut   = K1band * 9;		// grid number of k_para
double const	K1omega = 1.0/K1band;		// grid in k_para domain

int const		Itrmax = 1000000;			// max number of iteration
double const	Tol0   = 0.000001;			// criteria for convergence

// Physical constants

double const	Field   = Fmax*1.0/Iband;	// electric field
double const	Bloch   = Fmax*Domega;		// Bloch frequency

double const	Temp    = 0.000001*Domega;	// temperature (Zero Temp = 0.000001*Domega)
double const	Gamma   = 0.01;				// system-bath scattering rate
double const	A		= 0.6;				// off-diagonal dispersion (+, in unit 4D)
double const	B		= 0.6;				// spin-orbit coupling strength (+, in unit 4D)
double const	M		= -0.3;				// mass gap (+: trivial, -: nontrivial, in unit 4D)
double const	Vimp	= 0.08;				// Coupling strength to impurity (0, 0.05, 0.08)

// Define subroutine

double *SubLocGreenA (int rank, double *RetSelfA);
double *SubLocGreenB (int rank, double *RetSelfB);
double *SubSemiLocDos (int rank, double Kperp, double *RetSelfA, double *RetSelfB);


/************************************************************************************************/
/************************************** [Main routine] ******************************************/
/************************************************************************************************/

main(int argc, char **argv)
{
    // Open the saving file
    
    FILE *f1;
    f1 = fopen("HgTeZakA6B6Mm3E016I008","wt");
    
    // Initiate the MPI system
    
    int rank;		// Rank of my CPU
    int source;		// CPU sending message
    int dest = 0;	// CPU receiving message
    int root = 0;	// CPU broadcasting message to all the others
    int tag = 0;	// Indicator distinguishing messages
    
    MPI_Status status;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    // Declare variables
    
    int i, ik, ik1, iw, n;
    double Kperp, d1, d2, d3, ek, Ek1;
    double Alpha, TolA, CritA, Crit0A, TolB, CritB, Crit0B;
    double RetSelfA[2*Icut], OldRetSelfA[2*Icut], NewRetSelfA[2*Icut];
    double RetSelfB[2*Icut], OldRetSelfB[2*Icut], NewRetSelfB[2*Icut];
    double *LocGreenASub, RetLocGreenA[2*Icut], LocDosA[Icut], OldDosA[Icut];
    double *LocGreenBSub, RetLocGreenB[2*Icut], LocDosB[Icut], OldDosB[Icut];
    double *SemiLocDosSub, SemiLocDos[Icut];
    
    
    /************************************************************************************************/
    /******************************** Impurity self-energy via SCBA *********************************/
    /************************************************************************************************/
    
    if (rank == 0) {
        
        printf ("Step / Tol_LocDosA / Tol_LocDosB \n");
    }
    
    Crit0A = Crit0B = 0.0;
    
    for (iw = 0; iw <= Icut-1; iw++)
    {
        RE1(RetSelfA,iw) = IM1(RetSelfA,iw) = 0;
        OldDosA[iw] = 0.1;
        Crit0A += pow(OldDosA[iw], 2);
        
        RE1(RetSelfB,iw) = IM1(RetSelfB,iw) = 0;
        OldDosB[iw] = 0.1;
        Crit0B += pow(OldDosB[iw], 2);
    }
    
    for (i = 1; i <= Itrmax; i++) {
        
        // Local Green's function A
        
        if (rank >= 1 && rank <= Ncpu-1) {
            
            LocGreenASub = SubLocGreenA (rank, RetSelfA);
            MPI_Send (LocGreenASub, 2*Icut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            
        } else if (rank == 0) {
            
            for (source = 0; source <= Ncpu-1; source++)
            {
                if (source == 0) {
                    
                    LocGreenASub = SubLocGreenA (source, RetSelfA);
                    
                } else {
                    
                    MPI_Recv (LocGreenASub, 2*Icut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
                }
                
                for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++)
                {
                    for (n = 0; n <= Ncut-1; n++)
                    {
                        // Retarded local Green's function
                        
                        RE1(RetLocGreenA,iw+n*Fmax) = RE1(LocGreenASub,iw+Fmax*(n-source)/Ncpu);
                        IM1(RetLocGreenA,iw+n*Fmax) = IM1(LocGreenASub,iw+Fmax*(n-source)/Ncpu);
                    }
                }
            }
            
            for (iw = 0; iw <= Icut-1; iw++)
            {
                // Self-energy for impurity averaging (SCBA)
                
                RE1(RetSelfA,iw) = pow(Vimp,2) * RE1(RetLocGreenA,iw);
                IM1(RetSelfA,iw) = pow(Vimp,2) * IM1(RetLocGreenA,iw);
                
                // Linear Mixing (to enhance the convergence of solution)
                
                if (i == 1) {
                    
                    Alpha = 1.0;
                    
                } else {
                    
                    Alpha = 0.7;
                }
                
                RE1(NewRetSelfA,iw) = RE1(RetSelfA,iw);
                IM1(NewRetSelfA,iw) = IM1(RetSelfA,iw);
                
                RE1(RetSelfA,iw) = Alpha * RE1(RetSelfA,iw) + (1.0-Alpha) * RE1(OldRetSelfA,iw);
                IM1(RetSelfA,iw) = Alpha * IM1(RetSelfA,iw) + (1.0-Alpha) * IM1(OldRetSelfA,iw);
                
                RE1(OldRetSelfA,iw) = RE1(RetSelfA,iw);
                IM1(OldRetSelfA,iw) = IM1(RetSelfA,iw);
                
                // Local Dos
                
                LocDosA[iw] = - IM1(RetLocGreenA,iw) / M_PI;
            }
        }
        
        // Local Green's function B
        
        if (rank >= 1 && rank <= Ncpu-1) {
            
            LocGreenBSub = SubLocGreenB (rank, RetSelfB);
            MPI_Send (LocGreenBSub, 2*Icut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            
        } else if (rank == 0) {
            
            for (source = 0; source <= Ncpu-1; source++)
            {
                if (source == 0) {
                    
                    LocGreenBSub = SubLocGreenB (source, RetSelfB);
                    
                } else {
                    
                    MPI_Recv (LocGreenBSub, 2*Icut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
                }
                
                for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++)
                {
                    for (n = 0; n <= Ncut-1; n++)
                    {
                        // Retarded local Green's function
                        
                        RE1(RetLocGreenB,iw+n*Fmax) = RE1(LocGreenBSub,iw+Fmax*(n-source)/Ncpu);
                        IM1(RetLocGreenB,iw+n*Fmax) = IM1(LocGreenBSub,iw+Fmax*(n-source)/Ncpu);
                    }
                }
            }
            
            for (iw = 0; iw <= Icut-1; iw++)
            {
                // Self-energy for impurity averaging (SCBA)
                
                RE1(RetSelfB,iw) = pow(Vimp,2) * RE1(RetLocGreenB,iw);
                IM1(RetSelfB,iw) = pow(Vimp,2) * IM1(RetLocGreenB,iw);
                
                // Linear Mixing (to enhance the convergence of solution)
                
                if (i == 1) {
                    
                    Alpha = 1.0;
                    
                } else {
                    
                    Alpha = 0.7;
                }
                
                RE1(NewRetSelfB,iw) = RE1(RetSelfB,iw);
                IM1(NewRetSelfB,iw) = IM1(RetSelfB,iw);
                
                RE1(RetSelfB,iw) = Alpha * RE1(RetSelfB,iw) + (1.0-Alpha) * RE1(OldRetSelfB,iw);
                IM1(RetSelfB,iw) = Alpha * IM1(RetSelfB,iw) + (1.0-Alpha) * IM1(OldRetSelfB,iw);
                
                RE1(OldRetSelfB,iw) = RE1(RetSelfB,iw);
                IM1(OldRetSelfB,iw) = IM1(RetSelfB,iw);
                
                // Local Dos
                
                LocDosB[iw] = - IM1(RetLocGreenB,iw) / M_PI;
            }
        }
        
        // Broadcast data to the other CPUs
        
        MPI_Bcast (RetSelfA, 2*Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
        MPI_Bcast (LocDosA, Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
        
        MPI_Bcast (RetSelfB, 2*Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
        MPI_Bcast (LocDosB, Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
        
        // Convergence test
        
        CritA = CritB = 0.0;
        
        for (iw = 0; iw <= Icut-1; iw++)
        {
            CritA += pow(LocDosA[iw] - OldDosA[iw], 2);
            CritB += pow(LocDosB[iw] - OldDosB[iw], 2);
        }
        
        TolA = sqrt(CritA/Crit0A);
        TolB = sqrt(CritB/Crit0B);
        
        Crit0A = Crit0B = 0.0;
        
        for (iw = 0; iw <= Icut-1; iw++)
        {
            OldDosA[iw] = LocDosA[iw];
            Crit0A += pow(OldDosA[iw], 2);
            
            OldDosB[iw] = LocDosB[iw];
            Crit0B += pow(OldDosB[iw], 2);
        }
        
        if (rank == 0) {
            
            printf ("%d %e %e \n", i, TolA, TolB);
        }
        
        if (i > 1 && TolA < Tol0 && TolB < Tol0) {
            
            if (rank == 0) {
                
                printf (">> Converged within %e \n\n", Tol0);
            }
            
            break;
        }
    }
    
    
    /************************************************************************************************/
    /********************************** Zak phase (spin up + down) **********************************/
    /************************************************************************************************/
    
    // Loop for the lateral momentum
    
    for (ik = 0; ik <= Kcut-1; ik++)
    {
        // Set up the lateral momentum (spin up)
        
        Kperp = (ik - (Kcut-1.0)/2.0) * 2.0/Kcut * M_PI;
        
        // Semi-local Dos
        
        if (rank >= 1 && rank <= Ncpu-1) {
            
            SemiLocDosSub = SubSemiLocDos (rank, Kperp, RetSelfA, RetSelfB);
            MPI_Send (SemiLocDosSub, Icut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            
        } else if (rank == 0) {
            
            for (source = 0; source <= Ncpu-1; source++)
            {
                if (source == 0) {
                    
                    SemiLocDosSub = SubSemiLocDos (source, Kperp, RetSelfA, RetSelfB);
                    
                } else {
                    
                    MPI_Recv (SemiLocDosSub, Icut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
                }
                
                for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++)
                {
                    for (n = 0; n <= Ncut-1; n++)
                    {
                        SemiLocDos[iw+n*Fmax] = SemiLocDosSub[iw+Fmax*(n-source)/Ncpu];
                    }
                }
            }
        }
        
        // Set up the lateral momentum (spin down)
        
        Kperp = - (ik - (Kcut-1.0)/2.0) * 2.0/Kcut * M_PI;
        
        // Semi-local Dos
        
        if (rank >= 1 && rank <= Ncpu-1) {
            
            SemiLocDosSub = SubSemiLocDos (rank, Kperp, RetSelfA, RetSelfB);
            MPI_Send (SemiLocDosSub, Icut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            
        } else if (rank == 0) {
            
            for (source = 0; source <= Ncpu-1; source++)
            {
                if (source == 0) {
                    
                    SemiLocDosSub = SubSemiLocDos (source, Kperp, RetSelfA, RetSelfB);
                    
                } else {
                    
                    MPI_Recv (SemiLocDosSub, Icut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
                }
                
                for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++)
                {
                    for (n = 0; n <= Ncut-1; n++)
                    {
                        SemiLocDos[iw+n*Fmax] += SemiLocDosSub[iw+Fmax*(n-source)/Ncpu];
                    }
                }
            }
        }
        
        // Find the band center in equilibrium
        
        if (rank == 0) {
            
            Ek1 = 0;
            
            for (ik1 = 0; ik1 <= K1cut-1; ik1++)
            {
                if ( fabs((ik1-(K1cut-1.0)/2.0)*K1omega) <= M_PI ) {
                    
                    d1 = A * sin((ik1-(K1cut-1.0)/2.0)*K1omega);
                    d2 = - A * sin(Kperp);
                    d3 = M + 2.0 * B * (2.0 - cos((ik1-(K1cut-1.0)/2.0)*K1omega) - cos(Kperp));
                    ek = 0.5 * (2.0 - cos((ik1-(K1cut-1.0)/2.0)*K1omega) - cos(Kperp));
                    
                    Ek1 += K1omega/(2.0*M_PI) * (ek - sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2)));
                }
            }
        }
        
        // Save data on output files
        
        if (rank == 0) {
            
            for (iw = 0; iw <= Icut-1; iw++)
            {
                fprintf(f1, "%f %f %e\n",
                        (ik-(Kcut-1.0)/2.0)*2.0/Kcut,
                        (iw-(Icut-1.0)/2.0)*Domega - Ek1,
                        SemiLocDos[iw]);
            }
            
            fprintf(f1, "\n");
            
            printf("For the normalized Kperp = %f, band centers = %f \n",
                   (ik-(Kcut-1.0)/2.0)*2.0/Kcut, Ek1);
        }
    }
    
    // Close save file
    
    fclose(f1);
    
    // Finish MPI system
    
    MPI_Finalize();
}


/************************************************************************************************/
/***************************** [Subroutine] Local Green's function A ****************************/
/************************************************************************************************/

double *SubLocGreenA (int rank, double *RetSelfA)
{
    // Definition of local variables
    
    int ik, iw, n, p, q, s;
    double Kperp, FloHamRe;
    double OrbMixARe, OrbMixAIm, GInvARe, GInvAIm;
    static double LocGreenASub[2*Icut/Ncpu];
    
    for (iw = Fmax*rank/Ncpu; iw <= Fmax*(rank+1)/Ncpu-1; iw++)
    {
        for (n = 0; n <= Ncut-1; n++)
        {
            RE1(LocGreenASub,iw+Fmax*(n-rank)/Ncpu) = 0;
            IM1(LocGreenASub,iw+Fmax*(n-rank)/Ncpu) = 0;
            
            for (ik = 0; ik <= Kcut-1; ik++)
            {
                // Set up the lateral momentum
                
                Kperp = (ik-(Kcut-1.0)/2.0)*2.0/Kcut * M_PI;
                
                // 1-band retarded lattice Green's function
                
                gsl_matrix_complex *kGreenInv1B = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        if (p == q) {
                            
                            FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + M - 2.0 * (0.25 - B) * (2.0 - cos(Kperp));
                            gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0.25 - B, 0));
                            
                        } else {
                            
                            gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, 0));
                        }
                    }
                }
                
                // Matrix inversion (1 band)
                
                gsl_permutation *perm1B = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *kGreen1B = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (kGreenInv1B, perm1B, &s);
                gsl_linalg_complex_LU_invert (kGreenInv1B, perm1B, kGreen1B);
                
                // Free the previous allocation
                
                gsl_permutation_free (perm1B);
                gsl_matrix_complex_free (kGreenInv1B);
                
                // 2-band retarded lattice Green's function
                
                gsl_matrix_complex *kGreenInv2A = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_complex kGreenP1B, kGreenP2B, kGreenP3B, kGreenP4B, kGreenP5B, kGreenP6B, kGreenP7B, kGreenP8B, kGreenP9B;
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        kGreenP1B = gsl_matrix_complex_get (kGreen1B, p, q);
                        
                        if (q == 0) {
                            
                            kGreenP2B = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP2B = gsl_matrix_complex_get (kGreen1B, p, q-1);
                        }
                        
                        if (q == Ncut-1) {
                            
                            kGreenP3B = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP3B = gsl_matrix_complex_get (kGreen1B, p, q+1);
                        }
                        
                        if (p == 0) {
                            
                            kGreenP4B = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP4B = gsl_matrix_complex_get (kGreen1B, p-1, q);
                        }
                        
                        if (p == Ncut-1) {
                            
                            kGreenP5B = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP5B = gsl_matrix_complex_get (kGreen1B, p+1, q);
                        }
                        
                        if (p == Ncut-1 || q == Ncut-1) {
                            
                            kGreenP6B = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP6B = gsl_matrix_complex_get (kGreen1B, p+1, q+1);
                        }
                        
                        if (p == 0 || q == 0) {
                            
                            kGreenP7B = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP7B = gsl_matrix_complex_get (kGreen1B, p-1, q-1);
                        }
                        
                        if (p == Ncut-1 || q == 0) {
                            
                            kGreenP8B = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP8B = gsl_matrix_complex_get (kGreen1B, p+1, q-1);
                        }
                        
                        if (p == 0 || q == Ncut-1) {
                            
                            kGreenP9B = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP9B = gsl_matrix_complex_get (kGreen1B, p-1, q+1);
                        }
                        
                        OrbMixARe = pow(0.5*A,2) * ( pow(2.0*sin(Kperp),2) * GSL_REAL(kGreenP1B)
                                    - 2.0*sin(Kperp) * (GSL_REAL(kGreenP2B) - GSL_REAL(kGreenP3B)
                                    + GSL_REAL(kGreenP4B) - GSL_REAL(kGreenP5B))
                                    + GSL_REAL(kGreenP6B) + GSL_REAL(kGreenP7B)
                                    - GSL_REAL(kGreenP8B) - GSL_REAL(kGreenP9B) );
                        
                        OrbMixAIm = pow(0.5*A,2) * ( pow(2.0*sin(Kperp),2) * GSL_IMAG(kGreenP1B)
                                    - 2.0*sin(Kperp) * (GSL_IMAG(kGreenP2B) - GSL_IMAG(kGreenP3B)
                                    + GSL_IMAG(kGreenP4B) - GSL_IMAG(kGreenP5B))
                                    + GSL_IMAG(kGreenP6B) + GSL_IMAG(kGreenP7B)
                                    - GSL_IMAG(kGreenP8B) - GSL_IMAG(kGreenP9B) );
                        
                        if (p == q) {
                            
                            GInvARe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch - M - 2.0 * (0.25 + B) * (2.0 - cos(Kperp)) - OrbMixARe - RE1(RetSelfA,iw+p*Fmax);
                            GInvAIm = 0.5*Gamma - OrbMixAIm - IM1(RetSelfA,iw+p*Fmax);
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            GInvARe = 0.25 + B - OrbMixARe;
                            GInvAIm = - OrbMixAIm;
                            
                        } else {
                            
                            GInvARe = - OrbMixARe;
                            GInvAIm = - OrbMixAIm;
                        }
                        
                        gsl_matrix_complex_set (kGreenInv2A, p, q, gsl_complex_rect (GInvARe, GInvAIm));
                    }
                }
                
                // Free the previous allocation
                
                gsl_matrix_complex_free (kGreen1B);
                
                // Matrix inversion (2 bands)
                
                gsl_permutation *perm2A = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *kGreen2A = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (kGreenInv2A, perm2A, &s);
                gsl_linalg_complex_LU_invert (kGreenInv2A, perm2A, kGreen2A);
                
                // 2-band retarded local Green's function
                
                gsl_complex RetkGreenA = gsl_matrix_complex_get (kGreen2A, n, n);
                
                RE1(LocGreenASub,iw+Fmax*(n-rank)/Ncpu) += (1.0/Kcut) * GSL_REAL(RetkGreenA);
                IM1(LocGreenASub,iw+Fmax*(n-rank)/Ncpu) += (1.0/Kcut) * GSL_IMAG(RetkGreenA);
                
                // Free the previous allocation
                
                gsl_permutation_free (perm2A);
                gsl_matrix_complex_free (kGreen2A);
                gsl_matrix_complex_free (kGreenInv2A);
            }
        }
    }
    
    return LocGreenASub;
}


/************************************************************************************************/
/***************************** [Subroutine] Local Green's function B ****************************/
/************************************************************************************************/

double *SubLocGreenB (int rank, double *RetSelfB)
{
    // Definition of local variables
    
    int ik, iw, n, p, q, s;
    double Kperp, FloHamRe;
    double OrbMixBRe, OrbMixBIm, GInvBRe, GInvBIm;
    static double LocGreenBSub[2*Icut/Ncpu];
    
    for (iw = Fmax*rank/Ncpu; iw <= Fmax*(rank+1)/Ncpu-1; iw++)
    {
        for (n = 0; n <= Ncut-1; n++)
        {
            RE1(LocGreenBSub,iw+Fmax*(n-rank)/Ncpu) = 0;
            IM1(LocGreenBSub,iw+Fmax*(n-rank)/Ncpu) = 0;
            
            for (ik = 0; ik <= Kcut-1; ik++)
            {
                // Set up the lateral momentum
                
                Kperp = (ik-(Kcut-1.0)/2.0)*2.0/Kcut * M_PI;
                
                // 1-band retarded lattice Green's function
                
                gsl_matrix_complex *kGreenInv1A = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        if (p == q) {
                            
                            FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch - M - 2.0 * (0.25 + B) * (2.0 - cos(Kperp));
                            gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0.25 + B, 0));
                        
                        } else {
                            
                            gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, 0));
                        }
                    }
                }
                
                // Matrix inversion (1 band)
                
                gsl_permutation *perm1A = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *kGreen1A = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (kGreenInv1A, perm1A, &s);
                gsl_linalg_complex_LU_invert (kGreenInv1A, perm1A, kGreen1A);
                
                // Free the previous allocation
                
                gsl_permutation_free (perm1A);
                gsl_matrix_complex_free (kGreenInv1A);
                
                // 2-band retarded lattice Green's function
                
                gsl_matrix_complex *kGreenInv2B = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_complex kGreenP1A, kGreenP2A, kGreenP3A, kGreenP4A, kGreenP5A, kGreenP6A, kGreenP7A, kGreenP8A, kGreenP9A;
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        kGreenP1A = gsl_matrix_complex_get (kGreen1A, p, q);
                        
                        if (q == 0) {
                            
                            kGreenP2A = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP2A = gsl_matrix_complex_get (kGreen1A, p, q-1);
                        }
                        
                        if (q == Ncut-1) {
                            
                            kGreenP3A = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP3A = gsl_matrix_complex_get (kGreen1A, p, q+1);
                        }
                        
                        if (p == 0) {
                            
                            kGreenP4A = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP4A = gsl_matrix_complex_get (kGreen1A, p-1, q);
                        }
                        
                        if (p == Ncut-1) {
                            
                            kGreenP5A = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP5A = gsl_matrix_complex_get (kGreen1A, p+1, q);
                        }
                        
                        if (p == Ncut-1 || q == Ncut-1) {
                            
                            kGreenP6A = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP6A = gsl_matrix_complex_get (kGreen1A, p+1, q+1);
                        }
                        
                        if (p == 0 || q == 0) {
                            
                            kGreenP7A = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP7A = gsl_matrix_complex_get (kGreen1A, p-1, q-1);
                        }
                        
                        if (p == Ncut-1 || q == 0) {
                            
                            kGreenP8A = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP8A = gsl_matrix_complex_get (kGreen1A, p+1, q-1);
                        }
                        
                        if (p == 0 || q == Ncut-1) {
                            
                            kGreenP9A = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            kGreenP9A = gsl_matrix_complex_get (kGreen1A, p-1, q+1);
                        }
                        
                        OrbMixBRe = pow(0.5*A,2) * ( pow(2.0*sin(Kperp),2) * GSL_REAL(kGreenP1A)
                                    + 2.0*sin(Kperp) * (GSL_REAL(kGreenP2A) - GSL_REAL(kGreenP3A)
                                    + GSL_REAL(kGreenP4A) - GSL_REAL(kGreenP5A))
                                    + GSL_REAL(kGreenP6A) + GSL_REAL(kGreenP7A)
                                    - GSL_REAL(kGreenP8A) - GSL_REAL(kGreenP9A) );
                        
                        OrbMixBIm = pow(0.5*A,2) * ( pow(2.0*sin(Kperp),2) * GSL_IMAG(kGreenP1A)
                                    + 2.0*sin(Kperp) * (GSL_IMAG(kGreenP2A) - GSL_IMAG(kGreenP3A)
                                    + GSL_IMAG(kGreenP4A) - GSL_IMAG(kGreenP5A))
                                    + GSL_IMAG(kGreenP6A) + GSL_IMAG(kGreenP7A)
                                    - GSL_IMAG(kGreenP8A) - GSL_IMAG(kGreenP9A) );
                        
                        if (p == q) {
                            
                            GInvBRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + M - 2.0 * (0.25 - B) * (2.0 - cos(Kperp)) - OrbMixBRe - RE1(RetSelfB,iw+p*Fmax);
                            GInvBIm = 0.5*Gamma - OrbMixBIm - IM1(RetSelfB,iw+p*Fmax);
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            GInvBRe = 0.25 - B - OrbMixBRe;
                            GInvBIm = - OrbMixBIm;
                            
                        } else {
                            
                            GInvBRe = - OrbMixBRe;
                            GInvBIm = - OrbMixBIm;
                        }
                        
                        gsl_matrix_complex_set (kGreenInv2B, p, q, gsl_complex_rect (GInvBRe, GInvBIm));
                    }
                }
                
                // Free the previous allocation
                
                gsl_matrix_complex_free (kGreen1A);
                
                // Matrix inversion (2 bands)
                
                gsl_permutation *perm2B = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *kGreen2B = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (kGreenInv2B, perm2B, &s);
                gsl_linalg_complex_LU_invert (kGreenInv2B, perm2B, kGreen2B);
                
                // 2-band retarded local Green's function
                
                gsl_complex RetkGreenB = gsl_matrix_complex_get (kGreen2B, n, n);
                
                RE1(LocGreenBSub,iw+Fmax*(n-rank)/Ncpu) += (1.0/Kcut) * GSL_REAL(RetkGreenB);
                IM1(LocGreenBSub,iw+Fmax*(n-rank)/Ncpu) += (1.0/Kcut) * GSL_IMAG(RetkGreenB);
                
                // Free the previous allocation
                
                gsl_permutation_free (perm2B);
                gsl_matrix_complex_free (kGreen2B);
                gsl_matrix_complex_free (kGreenInv2B);
            }
        }
    }
    
    return LocGreenBSub;
}


/************************************************************************************************/
/***************************** [Subroutine] Semi-Local Dos (A + B) ******************************/
/************************************************************************************************/

double *SubSemiLocDos (int rank, double Kperp, double *RetSelfA, double *RetSelfB)
{
    // Definition of local variables
    
    int iw, n, p, q, s;
    double FloHamRe;
    double OrbMixARe, OrbMixAIm, GInvARe, GInvAIm;
    double OrbMixBRe, OrbMixBIm, GInvBRe, GInvBIm;
    static double SemiLocDosSub[Icut/Ncpu];
    
    // Find the semi-local Dos at low field
    
    for (iw = Fmax*rank/Ncpu; iw <= Fmax*(rank+1)/Ncpu-1; iw++)
    {
        for (n = 0; n <= Ncut-1; n++)
        {
            // 1-band retarded lattice Green's function
            
            gsl_matrix_complex *kGreenInv1A = gsl_matrix_complex_alloc (Ncut, Ncut);
            gsl_matrix_complex *kGreenInv1B = gsl_matrix_complex_alloc (Ncut, Ncut);
            
            for (p = 0; p <= Ncut-1; p++)
            {
                for (q = 0; q <= Ncut-1; q++)
                {
                    if (p == q) {
                        
                        FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                   - M - 2.0 * (0.25 + B) * (2.0 - cos(Kperp));
                        
                        gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
                        
                        FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                   + M - 2.0 * (0.25 - B) * (2.0 - cos(Kperp));
                        
                        gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
                        
                    } else if (p == q + 1) {
                        
                        gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0.25 + B, 0));
                        gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0.25 - B, 0));
                        
                    } else if (p == q - 1) {
                        
                        gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0.25 + B, 0));
                        gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0.25 - B, 0));
                        
                    } else {
                        
                        gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, 0));
                        gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, 0));
                    }
                }
            }
            
            // Matrix inversion (1 band)
            
            gsl_permutation *perm1A = gsl_permutation_alloc (Ncut);
            gsl_matrix_complex *kGreen1A = gsl_matrix_complex_alloc (Ncut, Ncut);
            gsl_linalg_complex_LU_decomp (kGreenInv1A, perm1A, &s);
            gsl_linalg_complex_LU_invert (kGreenInv1A, perm1A, kGreen1A);
            
            gsl_permutation *perm1B = gsl_permutation_alloc (Ncut);
            gsl_matrix_complex *kGreen1B = gsl_matrix_complex_alloc (Ncut, Ncut);
            gsl_linalg_complex_LU_decomp (kGreenInv1B, perm1B, &s);
            gsl_linalg_complex_LU_invert (kGreenInv1B, perm1B, kGreen1B);
            
            // Free the previous allocation
            
            gsl_permutation_free (perm1A);
            gsl_matrix_complex_free (kGreenInv1A);
            
            gsl_permutation_free (perm1B);
            gsl_matrix_complex_free (kGreenInv1B);
            
            // 2-band retarded lattice Green's function
            
            gsl_matrix_complex *kGreenInv2A = gsl_matrix_complex_alloc (Ncut, Ncut);
            gsl_complex kGreenP1A, kGreenP2A, kGreenP3A;
            gsl_complex kGreenP4A, kGreenP5A, kGreenP6A;
            gsl_complex kGreenP7A, kGreenP8A, kGreenP9A;
            
            gsl_matrix_complex *kGreenInv2B = gsl_matrix_complex_alloc (Ncut, Ncut);
            gsl_complex kGreenP1B, kGreenP2B, kGreenP3B;
            gsl_complex kGreenP4B, kGreenP5B, kGreenP6B;
            gsl_complex kGreenP7B, kGreenP8B, kGreenP9B;
            
            for (p = 0; p <= Ncut-1; p++)
            {
                for (q = 0; q <= Ncut-1; q++)
                {
                    kGreenP1B = gsl_matrix_complex_get (kGreen1B, p, q);
                    
                    if (q == 0) {
                        
                        kGreenP2B = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP2B = gsl_matrix_complex_get (kGreen1B, p, q-1);
                    }
                    
                    if (q == Ncut-1) {
                        
                        kGreenP3B = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP3B = gsl_matrix_complex_get (kGreen1B, p, q+1);
                    }
                    
                    if (p == 0) {
                        
                        kGreenP4B = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP4B = gsl_matrix_complex_get (kGreen1B, p-1, q);
                    }
                    
                    if (p == Ncut-1) {
                        
                        kGreenP5B = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP5B = gsl_matrix_complex_get (kGreen1B, p+1, q);
                    }
                    
                    if (p == Ncut-1 || q == Ncut-1) {
                        
                        kGreenP6B = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP6B = gsl_matrix_complex_get (kGreen1B, p+1, q+1);
                    }
                    
                    if (p == 0 || q == 0) {
                        
                        kGreenP7B = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP7B = gsl_matrix_complex_get (kGreen1B, p-1, q-1);
                    }
                    
                    if (p == Ncut-1 || q == 0) {
                        
                        kGreenP8B = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP8B = gsl_matrix_complex_get (kGreen1B, p+1, q-1);
                    }
                    
                    if (p == 0 || q == Ncut-1) {
                        
                        kGreenP9B = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP9B = gsl_matrix_complex_get (kGreen1B, p-1, q+1);
                    }
                    
                    OrbMixARe = pow(0.5*A,2) * ( pow(2.0*sin(Kperp),2) * GSL_REAL(kGreenP1B)
                                - 2.0*sin(Kperp) * (GSL_REAL(kGreenP2B) - GSL_REAL(kGreenP3B)
                                + GSL_REAL(kGreenP4B) - GSL_REAL(kGreenP5B))
                                + GSL_REAL(kGreenP6B) + GSL_REAL(kGreenP7B)
                                - GSL_REAL(kGreenP8B) - GSL_REAL(kGreenP9B) );
                    
                    OrbMixAIm = pow(0.5*A,2) * ( pow(2.0*sin(Kperp),2) * GSL_IMAG(kGreenP1B)
                                - 2.0*sin(Kperp) * (GSL_IMAG(kGreenP2B) - GSL_IMAG(kGreenP3B)
                                + GSL_IMAG(kGreenP4B) - GSL_IMAG(kGreenP5B))
                                + GSL_IMAG(kGreenP6B) + GSL_IMAG(kGreenP7B)
                                - GSL_IMAG(kGreenP8B) - GSL_IMAG(kGreenP9B) );
                    
                    kGreenP1A = gsl_matrix_complex_get (kGreen1A, p, q);
                    
                    if (q == 0) {
                        
                        kGreenP2A = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP2A = gsl_matrix_complex_get (kGreen1A, p, q-1);
                    }
                    
                    if (q == Ncut-1) {
                        
                        kGreenP3A = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP3A = gsl_matrix_complex_get (kGreen1A, p, q+1);
                    }
                    
                    if (p == 0) {
                        
                        kGreenP4A = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP4A = gsl_matrix_complex_get (kGreen1A, p-1, q);
                    }
                    
                    if (p == Ncut-1) {
                        
                        kGreenP5A = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP5A = gsl_matrix_complex_get (kGreen1A, p+1, q);
                    }
                    
                    if (p == Ncut-1 || q == Ncut-1) {
                        
                        kGreenP6A = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP6A = gsl_matrix_complex_get (kGreen1A, p+1, q+1);
                    }
                    
                    if (p == 0 || q == 0) {
                        
                        kGreenP7A = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP7A = gsl_matrix_complex_get (kGreen1A, p-1, q-1);
                    }
                    
                    if (p == Ncut-1 || q == 0) {
                        
                        kGreenP8A = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP8A = gsl_matrix_complex_get (kGreen1A, p+1, q-1);
                    }
                    
                    if (p == 0 || q == Ncut-1) {
                        
                        kGreenP9A = gsl_complex_rect (0, 0);
                        
                    } else {
                        
                        kGreenP9A = gsl_matrix_complex_get (kGreen1A, p-1, q+1);
                    }
                    
                    OrbMixBRe = pow(0.5*A,2) * ( pow(2.0*sin(Kperp),2) * GSL_REAL(kGreenP1A)
                                + 2.0*sin(Kperp) * (GSL_REAL(kGreenP2A) - GSL_REAL(kGreenP3A)
                                + GSL_REAL(kGreenP4A) - GSL_REAL(kGreenP5A))
                                + GSL_REAL(kGreenP6A) + GSL_REAL(kGreenP7A)
                                - GSL_REAL(kGreenP8A) - GSL_REAL(kGreenP9A) );
                    
                    OrbMixBIm = pow(0.5*A,2) * ( pow(2.0*sin(Kperp),2) * GSL_IMAG(kGreenP1A)
                                + 2.0*sin(Kperp) * (GSL_IMAG(kGreenP2A) - GSL_IMAG(kGreenP3A)
                                + GSL_IMAG(kGreenP4A) - GSL_IMAG(kGreenP5A))
                                + GSL_IMAG(kGreenP6A) + GSL_IMAG(kGreenP7A)
                                - GSL_IMAG(kGreenP8A) - GSL_IMAG(kGreenP9A) );
                    
                    if (p == q) {
                        
                        GInvARe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch - M - 2.0 * (0.25 + B) * (2.0 - cos(Kperp)) - OrbMixARe - RE1(RetSelfA,iw+p*Fmax);
                        GInvAIm = 0.5*Gamma - OrbMixAIm - IM1(RetSelfA,iw+p*Fmax);
                        
                        GInvBRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + M - 2.0 * (0.25 - B) * (2.0 - cos(Kperp)) - OrbMixBRe - RE1(RetSelfB,iw+p*Fmax);
                        GInvBIm = 0.5*Gamma - OrbMixBIm - IM1(RetSelfB,iw+p*Fmax);
                        
                    } else if (p == q + 1 || p == q - 1) {
                        
                        GInvARe = 0.25 + B - OrbMixARe;
                        GInvAIm = - OrbMixAIm;
                        
                        GInvBRe = 0.25 - B - OrbMixBRe;
                        GInvBIm = - OrbMixBIm;
                        
                    } else {
                        
                        GInvARe = - OrbMixARe;
                        GInvAIm = - OrbMixAIm;
                        
                        GInvBRe = - OrbMixBRe;
                        GInvBIm = - OrbMixBIm;
                    }
                    
                    gsl_matrix_complex_set (kGreenInv2A, p, q, gsl_complex_rect (GInvARe, GInvAIm));
                    gsl_matrix_complex_set (kGreenInv2B, p, q, gsl_complex_rect (GInvBRe, GInvBIm));
                }
            }
            
            // Free the previous allocation
            
            gsl_matrix_complex_free (kGreen1A);
            gsl_matrix_complex_free (kGreen1B);
            
            // Matrix inversion (2 bands)
            
            gsl_permutation *perm2A = gsl_permutation_alloc (Ncut);
            gsl_matrix_complex *kGreen2A = gsl_matrix_complex_alloc (Ncut, Ncut);
            gsl_linalg_complex_LU_decomp (kGreenInv2A, perm2A, &s);
            gsl_linalg_complex_LU_invert (kGreenInv2A, perm2A, kGreen2A);
            
            gsl_permutation *perm2B = gsl_permutation_alloc (Ncut);
            gsl_matrix_complex *kGreen2B = gsl_matrix_complex_alloc (Ncut, Ncut);
            gsl_linalg_complex_LU_decomp (kGreenInv2B, perm2B, &s);
            gsl_linalg_complex_LU_invert (kGreenInv2B, perm2B, kGreen2B);
            
            // 2-band retarded local Green's function
            
            gsl_complex kGreenA = gsl_matrix_complex_get (kGreen2A, n, n);
            gsl_complex kGreenB = gsl_matrix_complex_get (kGreen2B, n, n);
            
            // Semi-local Dos
            
            SemiLocDosSub[iw+Fmax*(n-rank)/Ncpu] = - (GSL_IMAG(kGreenA) + GSL_IMAG(kGreenB)) / M_PI;
            
            // Free the previous allocation
            
            gsl_permutation_free (perm2A);
            gsl_matrix_complex_free (kGreen2A);
            gsl_matrix_complex_free (kGreenInv2A);
            
            gsl_permutation_free (perm2B);
            gsl_matrix_complex_free (kGreen2B);
            gsl_matrix_complex_free (kGreenInv2B);
        }
    }
    
    return SemiLocDosSub;
}

