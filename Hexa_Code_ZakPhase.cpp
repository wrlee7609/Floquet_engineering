/************************************************************************************************/
/************* [System] 2D Bloch oscillator (Hexagonal lattice)					  ***************/
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
#define RE2(z,i,j)((z)[(i)][2*(j)])
#define IM2(z,i,j)((z)[(i)][2*(j)+1])
 
// Numerical constants

int const		Mod     = 1;					// (1) Armchair; (2) Zigzag

double const	ScaFac  = 0.5;                  // Scaling factor
                                                // (1) 0.5; (2) 0.5*sqrt(3.0)

// Numerical constants

int	const		Ncpu    = 15;					// total number of CPUs

int const		Iband   = Ncpu * 45;			// grid number of frequency domain in a bandwidth
int const		Ncut    = 261;					// grid number of Floquet mode
                                                // (1) 261; (2) 149
int const		Fmax    = Ncpu * 7;				// Field/Domega
int const		Icut    = Ncut * Fmax;			// grid number of frequency domain in a full range
double const	Domega  = ScaFac/Iband;			// grid in frequency domain

int const		Kcut    = Ncpu * 7;				// grid number of k_perp

int const		K1band	= Ncpu * 35;			// grid number of k_para within 1 B.W.
int const		K1cut   = K1band * 9;			// grid number of k_para
double const	K1omega = 1.0/K1band;			// grid in k_para domain

// Physical constants

double const	Field   = Fmax*1.0/Iband;		// electric field
double const	Bloch   = Fmax*Domega;			// Bloch frequency

double const	Temp    = 0.000001*Domega;		// temperature (Zero Temp = 0.000001*Domega)
double const	Gamma   = 0.01;					// system-bath scattering rate
double const	Lsoc    = 0.0;					// spin-orbit coupling strength
                                                // 0, 0.1, 0.3
double const	Gap     = 0.0*sqrt(3.0)*Lsoc;   // (half) mass gap
                                                // 0, 2.0*sqrt(3.0)*Lsoc, 4.0*sqrt(3.0)*Lsoc

// Define subroutine

double *SubSemiLocDosUpA (int rank, double EkUp, double Kperp);
double *SubSemiLocDosDnA (int rank, double EkDn, double Kperp);
double *SubSemiLocDosUpB (int rank, double EkUp, double Kperp);
double *SubSemiLocDosDnB (int rank, double EkDn, double Kperp);


/************************************************************************************************/
/************************************** [Main routine] ******************************************/
/************************************************************************************************/

main(int argc, char **argv)
{	
	// Open the saving file
	
	FILE *f1;
	f1 = fopen("HexaZakG00L00AcE016","wt");
	
	// Initiate the MPI system
	
	int rank;		// Rank of my CPU
	int source;		// CPU sending message
	int dest = 0;	// CPU receiving message
	int tag = 0;	// Indicator distinguishing messages
	
	MPI_Status status;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	
	// Declare variables
	
	int ik, ik1, iw, n;
	double Kperp, d12, d3, EkUp, EkDn, Multi;
    double *SemiLocDosUpASub, SemiLocDosUpA[Icut];
    double *SemiLocDosDnASub, SemiLocDosDnA[Icut];
    double *SemiLocDosUpBSub, SemiLocDosUpB[Icut];
    double *SemiLocDosDnBSub, SemiLocDosDnB[Icut];
	
	// Loop for the lateral momentum
	
	for (ik = 0; ik <= Kcut-1; ik++)
	{
        // Set up the lateral momentum
		
		if (Mod == 1) {
			
			Kperp = (ik - (Kcut-1.0)/2.0) * 2.0/Kcut * M_PI/sqrt(3.0);
		} 
		
		if (Mod == 2) {
			
			Kperp = (ik - (Kcut-1.0)/2.0) * 2.0/Kcut * M_PI/3.0;
		}
		
        // Find the band center in equilibrium
        
        EkUp = EkDn = 0;
        
        for (ik1 = 0; ik1 <= K1cut-1; ik1++)
        {
            // Mod 1 : Armchair
            
            if ( Mod == 1 && fabs((ik1-(K1cut-1.0)/2.0)*K1omega) <= 2.0*M_PI/3.0 ) {
                
                d12 = sqrt(4.0*cos(1.5*(ik1-(K1cut-1.0)/2.0)*K1omega)*cos(0.5*sqrt(3.0)*Kperp) + 2.0*cos(sqrt(3.0)*Kperp) + 3.0);
                d3 = - Gap + Lsoc * (4.0*cos(1.5*(ik1-(K1cut-1.0)/2.0)*K1omega) * sin(0.5*sqrt(3.0)*Kperp) - 2.0*sin(sqrt(3.0)*Kperp));
                EkUp += - K1omega/(4.0*M_PI/3.0) * sqrt(pow(d12,2) + pow(d3,2));
                
                d12 = sqrt(4.0*cos(1.5*(ik1-(K1cut-1.0)/2.0)*K1omega)*cos(0.5*sqrt(3.0)*Kperp) + 2.0*cos(sqrt(3.0)*Kperp) + 3.0);
                d3 = - Gap - Lsoc * (4.0*cos(1.5*(ik1-(K1cut-1.0)/2.0)*K1omega) * sin(0.5*sqrt(3.0)*Kperp) - 2.0*sin(sqrt(3.0)*Kperp));
                EkDn += - K1omega/(4.0*M_PI/3.0) * sqrt(pow(d12,2) + pow(d3,2));
            }
            
            // Mod 2 : Zigzag
            
            if ( Mod == 2 && fabs((ik1-(K1cut-1.0)/2.0)*K1omega) <= 2.0*M_PI/sqrt(3.0) ) {
                
                d12 = sqrt(4.0*cos(1.5*Kperp)*cos(0.5*sqrt(3.0)*(ik1-(K1cut-1.0)/2.0)*K1omega) + 2.0*cos(sqrt(3.0)*(ik1-(K1cut-1.0)/2.0)*K1omega) + 3.0);
                d3 = - Gap + Lsoc * (4.0*cos(1.5*Kperp)*sin(0.5*sqrt(3.0)*(ik1-(K1cut-1.0)/2.0)*K1omega) - 2.0*sin(sqrt(3.0)*(ik1-(K1cut-1.0)/2.0)*K1omega));
                EkUp += - K1omega/(4.0*M_PI/sqrt(3.0)) * sqrt(pow(d12,2) + pow(d3,2));
                
                d12 = sqrt(4.0*cos(1.5*Kperp)*cos(0.5*sqrt(3.0)*(ik1-(K1cut-1.0)/2.0)*K1omega) + 2.0*cos(sqrt(3.0)*(ik1-(K1cut-1.0)/2.0)*K1omega) + 3.0);
                d3 = - Gap - Lsoc * (4.0*cos(1.5*Kperp)*sin(0.5*sqrt(3.0)*(ik1-(K1cut-1.0)/2.0)*K1omega) - 2.0*sin(sqrt(3.0)*(ik1-(K1cut-1.0)/2.0)*K1omega));
                EkDn += - K1omega/(4.0*M_PI/sqrt(3.0)) * sqrt(pow(d12,2) + pow(d3,2));
            }
        }
        
		// Find the semi-local Dos (Up, A)
		
		if (rank >= 1 && rank <= Ncpu-1) {
			
			SemiLocDosUpASub = SubSemiLocDosUpA (rank, EkUp, Kperp);
			MPI_Send (SemiLocDosUpASub, Icut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
			
		} else if (rank == 0) {
			
			for (source = 0; source <= Ncpu-1; source++)
			{
				if (source == 0) {
					
					SemiLocDosUpASub = SubSemiLocDosUpA (source, EkUp, Kperp);
					
				} else {
					
					MPI_Recv (SemiLocDosUpASub, Icut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
				}
				
				for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++) 
				{
					for (n = 0; n <= Ncut-1; n++)
					{
						SemiLocDosUpA[iw+n*Fmax] = SemiLocDosUpASub[iw+Fmax*(n-source)/Ncpu];
					}
				}
			}
		}
        
        // Find the semi-local Dos (Dn, A)
        
        if (rank >= 1 && rank <= Ncpu-1) {
            
            SemiLocDosDnASub = SubSemiLocDosDnA (rank, EkDn, Kperp);
            MPI_Send (SemiLocDosDnASub, Icut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            
        } else if (rank == 0) {
            
            for (source = 0; source <= Ncpu-1; source++)
            {
                if (source == 0) {
                    
                    SemiLocDosDnASub = SubSemiLocDosDnA (source, EkDn, Kperp);
                    
                } else {
                    
                    MPI_Recv (SemiLocDosDnASub, Icut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
                }
                
                for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++) 
                {
                    for (n = 0; n <= Ncut-1; n++)
                    {
                        SemiLocDosDnA[iw+n*Fmax] = SemiLocDosDnASub[iw+Fmax*(n-source)/Ncpu];
                    }
                }
            }
        }
        
        // Find the semi-local Dos (Up, B)
        
        if (rank >= 1 && rank <= Ncpu-1) {
            
            SemiLocDosUpBSub = SubSemiLocDosUpB (rank, EkUp, Kperp);
            MPI_Send (SemiLocDosUpBSub, Icut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            
        } else if (rank == 0) {
            
            for (source = 0; source <= Ncpu-1; source++)
            {
                if (source == 0) {
                    
                    SemiLocDosUpBSub = SubSemiLocDosUpB (source, EkUp, Kperp);
                    
                } else {
                    
                    MPI_Recv (SemiLocDosUpBSub, Icut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
                }
                
                for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++)
                {
                    for (n = 0; n <= Ncut-1; n++)
                    {
                        SemiLocDosUpB[iw+n*Fmax] = SemiLocDosUpBSub[iw+Fmax*(n-source)/Ncpu];
                    }
                }
            }
        }
        
        // Find the semi-local Dos (Dn, B)
        
        if (rank >= 1 && rank <= Ncpu-1) {
            
            SemiLocDosDnBSub = SubSemiLocDosDnB (rank, EkDn, Kperp);
            MPI_Send (SemiLocDosDnBSub, Icut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
            
        } else if (rank == 0) {
            
            for (source = 0; source <= Ncpu-1; source++)
            {
                if (source == 0) {
                    
                    SemiLocDosDnBSub = SubSemiLocDosDnB (source, EkDn, Kperp);
                    
                } else {
                    
                    MPI_Recv (SemiLocDosDnBSub, Icut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
                }
                
                for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++)
                {
                    for (n = 0; n <= Ncut-1; n++)
                    {
                        SemiLocDosDnB[iw+n*Fmax] = SemiLocDosDnBSub[iw+Fmax*(n-source)/Ncpu];
                    }
                }
            }
        }
		
		// Save data on output files
		
		if (rank == 0) {
			
            if (Mod == 1) {
                
                Multi = 3.0;
            }
            
            if (Mod == 2) {
                
                Multi = 1.0;
            }
            
            for (n = (Ncut-1)/2 - 6; n <= (Ncut-1)/2 + 6; n++)
            {
                for (iw = 0; iw <= Fmax-1; iw++)
                {
                    fprintf(f1, "%f %f %e %e %e %e\n",
                            (ik-(Kcut-1.0)/2.0)*2.0/Kcut,
                            ((iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0)*Bloch)/Multi/Bloch,
                            SemiLocDosUpA[iw+n*Fmax],
                            SemiLocDosDnA[iw+n*Fmax],
                            SemiLocDosUpB[iw+n*Fmax],
                            SemiLocDosDnB[iw+n*Fmax]);
                }
            }
            
			fprintf(f1, "\n");
			
			printf("Complete the calculation for the normalized Kperp = %f, %f, %f\n",
				   (ik-(Kcut-1.0)/2.0)*2.0/Kcut, EkUp, EkDn);
		}
	}
		
	// Close save file
	
	fclose(f1);
	
	// Finish MPI system
	
	MPI_Finalize();
}


/************************************************************************************************/
/***************************** [Subroutine] Semi-Local Dos (Up, A) ******************************/
/************************************************************************************************/

double *SubSemiLocDosUpA (int rank, double EkUp, double Kperp)
{
    // Definition of local variables
    
    int iw, n, p, q, s;
    double FloHamRe, FloHamIm;
    double OrbMixARe, OrbMixAIm, GInvARe, GInvAIm;
    static double SemiLocDosUpASub[Icut/Ncpu];
    
    // Find the semi-local Dos at low field
    
    for (iw = Fmax*rank/Ncpu; iw <= Fmax*(rank+1)/Ncpu-1; iw++)
    {
        for (n = 0; n <= Ncut-1; n++)
        {
            if (n >= (Ncut-1)/2 - 6 && n <= (Ncut-1)/2 + 6) {
                
                // 1-band retarded lattice Green's function
                
                gsl_matrix_complex *kGreenInv1B = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        // Mod 1 : Armchair
                        
                        if (Mod == 1) {
                            
                            if (p == q) {
                                
                                FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkUp - Gap - 2.0*Lsoc * sin(sqrt(3.0)*Kperp);
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
                                
                            } else if (p == q + 3 || p == q - 3) {
                                
                                FloHamRe = 2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp);
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (FloHamRe, 0));
                                
                            } else {
                                
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, 0));
                            }
                        }
                        
                        // Mod 2 : Zigzag
                        
                        if (Mod == 2) {
                            
                            if (p == q) {
                                
                                FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkUp - Gap;
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
                                
                            } else if (p == q + 1) {
                                
                                FloHamIm = - 2.0*Lsoc * cos(1.5*Kperp);
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, FloHamIm));
                                
                            } else if (p == q - 1) {
                                
                                FloHamIm = 2.0*Lsoc * cos(1.5*Kperp);
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, FloHamIm));
                                
                            } else if (p == q + 2) {
                                
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, Lsoc));
                                
                            } else if (p == q - 2) {
                                
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, -Lsoc));
                                
                            } else {
                                
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, 0));
                            }
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
                        // Mod 1 : Armchair
                        
                        if (Mod == 1) {
                            
                            if (p >= Ncut-2 || q >= Ncut-2) {
                                
                                kGreenP1B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP1B = gsl_matrix_complex_get (kGreen1B, p+2, q+2);
                            }
                            
                            if (p == 0 || q == 0) {
                                
                                kGreenP2B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP2B = gsl_matrix_complex_get (kGreen1B, p-1, q-1);
                            }
                            
                            if (p == 0 || q >= Ncut-2) {
                                
                                kGreenP3B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP3B = gsl_matrix_complex_get (kGreen1B, p-1, q+2);
                            }
                            
                            if (p >= Ncut-2 || q == 0) {
                                
                                kGreenP4B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP4B = gsl_matrix_complex_get (kGreen1B, p+2, q-1);
                            }
                            
                            OrbMixARe = GSL_REAL(kGreenP1B) + pow(2.0*cos(0.5*sqrt(3.0)*Kperp),2) * GSL_REAL(kGreenP2B)
                                        + 2.0*cos(0.5*sqrt(3.0)*Kperp) * (GSL_REAL(kGreenP3B) + GSL_REAL(kGreenP4B));
                            
                            OrbMixAIm = GSL_IMAG(kGreenP1B) + pow(2.0*cos(0.5*sqrt(3.0)*Kperp),2) * GSL_IMAG(kGreenP2B)
                                        + 2.0*cos(0.5*sqrt(3.0)*Kperp) * (GSL_IMAG(kGreenP3B) + GSL_IMAG(kGreenP4B));
                            
                            if (p == q) {
                                
                                GInvARe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkUp + Gap + 2.0*Lsoc * sin(sqrt(3.0)*Kperp) - OrbMixARe;
                                GInvAIm = 0.5*Gamma - OrbMixAIm;
                                
                            } else if (p == q + 3 || p == q - 3) {
                                
                                GInvARe = - 2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp) - OrbMixARe;
                                GInvAIm = - OrbMixAIm;
                                
                            } else {
                                
                                GInvARe = - OrbMixARe;
                                GInvAIm = - OrbMixAIm;
                            }
                            
                            gsl_matrix_complex_set (kGreenInv2A, p, q, gsl_complex_rect (GInvARe, GInvAIm));
                        }
                        
                        // Mod 2 : Zigzag
                        
                        if (Mod == 2) {
                            
                            if (p == 0 || q == 0) {
                                
                                kGreenP1B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP1B = gsl_matrix_complex_get (kGreen1B, p-1, q-1);
                            }
                            
                            kGreenP2B = gsl_matrix_complex_get (kGreen1B, p, q);
                            
                            if (p == Ncut-1 || q == Ncut-1) {
                                
                                kGreenP3B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP3B = gsl_matrix_complex_get (kGreen1B, p+1, q+1);
                            }
                            
                            if (p == 0 || q == Ncut-1) {
                                
                                kGreenP4B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP4B = gsl_matrix_complex_get (kGreen1B, p-1, q+1);
                            }
                            
                            if (p == Ncut-1 || q == 0) {
                                
                                kGreenP5B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP5B = gsl_matrix_complex_get (kGreen1B, p+1, q-1);
                            }
                            
                            if (q == 0) {
                                
                                kGreenP6B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP6B = gsl_matrix_complex_get (kGreen1B, p, q-1);
                            }
                            
                            if (q == Ncut-1) {
                                
                                kGreenP7B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP7B = gsl_matrix_complex_get (kGreen1B, p, q+1);
                            }
                            
                            if (p == 0) {
                                
                                kGreenP8B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP8B = gsl_matrix_complex_get (kGreen1B, p-1, q);
                            }
                            
                            if (p == Ncut-1) {
                                
                                kGreenP9B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP9B = gsl_matrix_complex_get (kGreen1B, p+1, q);
                            }
                            
                            OrbMixARe = GSL_REAL(kGreenP1B) + GSL_REAL(kGreenP2B) + GSL_REAL(kGreenP3B) + GSL_REAL(kGreenP4B) + GSL_REAL(kGreenP5B)
                                        + cos(1.5*Kperp) * (GSL_REAL(kGreenP6B) + GSL_REAL(kGreenP7B)) - sin(1.5*Kperp) * (GSL_IMAG(kGreenP6B) + GSL_IMAG(kGreenP7B))
                                        + cos(1.5*Kperp) * (GSL_REAL(kGreenP8B) + GSL_REAL(kGreenP9B)) + sin(1.5*Kperp) * (GSL_IMAG(kGreenP8B) + GSL_IMAG(kGreenP9B));
                            
                            OrbMixAIm = GSL_IMAG(kGreenP1B) + GSL_IMAG(kGreenP2B) + GSL_IMAG(kGreenP3B) + GSL_IMAG(kGreenP4B) + GSL_IMAG(kGreenP5B)
                                        + cos(1.5*Kperp) * (GSL_IMAG(kGreenP6B) + GSL_IMAG(kGreenP7B)) + sin(1.5*Kperp) * (GSL_REAL(kGreenP6B) + GSL_REAL(kGreenP7B))
                                        + cos(1.5*Kperp) * (GSL_IMAG(kGreenP8B) + GSL_IMAG(kGreenP9B)) - sin(1.5*Kperp) * (GSL_REAL(kGreenP8B) + GSL_REAL(kGreenP9B));
                            
                            if (p == q) {
                                
                                GInvARe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkUp + Gap - OrbMixARe;
                                GInvAIm = 0.5*Gamma - OrbMixAIm;
                                
                            } else if (p == q + 1) {
                                
                                GInvARe = - OrbMixARe;
                                GInvAIm = 2.0*Lsoc * cos(1.5*Kperp) - OrbMixAIm;
                                
                            } else if (p == q - 1) {
                                
                                GInvARe = - OrbMixARe;
                                GInvAIm = - 2.0*Lsoc * cos(1.5*Kperp) - OrbMixAIm;
                                
                            } else if (p == q + 2) {
                                
                                GInvARe = - OrbMixARe;
                                GInvAIm = - Lsoc - OrbMixAIm;
                                
                            } else if (p == q - 2) {
                                
                                GInvARe = - OrbMixARe;
                                GInvAIm = Lsoc - OrbMixAIm;
                                
                            } else {
                                
                                GInvARe = - OrbMixARe;
                                GInvAIm = - OrbMixAIm;
                            }
                            
                            gsl_matrix_complex_set (kGreenInv2A, p, q, gsl_complex_rect (GInvARe, GInvAIm));
                        }
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
                
                gsl_complex kGreenA = gsl_matrix_complex_get (kGreen2A, n, n);
                
                // Semi-local Dos
                
                SemiLocDosUpASub[iw+Fmax*(n-rank)/Ncpu] = - GSL_IMAG(kGreenA) / M_PI;
                
                // Free the previous allocation
                
                gsl_permutation_free (perm2A);
                gsl_matrix_complex_free (kGreen2A);
                gsl_matrix_complex_free (kGreenInv2A);
                
            } else {
                
                // Semi-local Dos
                
                SemiLocDosUpASub[iw+Fmax*(n-rank)/Ncpu] = 0;
            }
        }
    }
    
    return SemiLocDosUpASub;
}


/************************************************************************************************/
/***************************** [Subroutine] Semi-Local Dos (Dn, A) ******************************/
/************************************************************************************************/

double *SubSemiLocDosDnA (int rank, double EkDn, double Kperp)
{
    // Definition of local variables
    
    int iw, n, p, q, s;
    double FloHamRe, FloHamIm;
    double OrbMixARe, OrbMixAIm, GInvARe, GInvAIm;
    static double SemiLocDosDnASub[Icut/Ncpu];
    
    // Find the semi-local Dos at low field
    
    for (iw = Fmax*rank/Ncpu; iw <= Fmax*(rank+1)/Ncpu-1; iw++)
    {
        for (n = 0; n <= Ncut-1; n++)
        {
            if (n >= (Ncut-1)/2 - 6 && n <= (Ncut-1)/2 + 6) {
                
                // 1-band retarded lattice Green's function
                
                gsl_matrix_complex *kGreenInv1B = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        // Mod 1 : Armchair
                        
                        if (Mod == 1) {
                            
                            if (p == q) {
                                
                                FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkDn - Gap + 2.0*Lsoc * sin(sqrt(3.0)*Kperp);
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
                                
                            } else if (p == q + 3 || p == q - 3) {
                                
                                FloHamRe = - 2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp);
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (FloHamRe, 0));
                                
                            } else {
                                
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, 0));
                            }
                        }
                        
                        // Mod 2 : Zigzag
                        
                        if (Mod == 2) {
                            
                            if (p == q) {
                                
                                FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkDn - Gap;
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
                                
                            } else if (p == q + 1) {
                                
                                FloHamIm = 2.0*Lsoc * cos(1.5*Kperp);
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, FloHamIm));
                                
                            } else if (p == q - 1) {
                                
                                FloHamIm = - 2.0*Lsoc * cos(1.5*Kperp);
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, FloHamIm));
                                
                            } else if (p == q + 2) {
                                
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, - Lsoc));
                                
                            } else if (p == q - 2) {
                                
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, Lsoc));
                                
                            } else {
                                
                                gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, 0));
                            }
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
                        // Mod 1 : Armchair
                        
                        if (Mod == 1) {
                            
                            if (p >= Ncut-2 || q >= Ncut-2) {
                                
                                kGreenP1B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP1B = gsl_matrix_complex_get (kGreen1B, p+2, q+2);
                            }
                            
                            if (p == 0 || q == 0) {
                                
                                kGreenP2B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP2B = gsl_matrix_complex_get (kGreen1B, p-1, q-1);
                            }
                            
                            if (p == 0 || q >= Ncut-2) {
                                
                                kGreenP3B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP3B = gsl_matrix_complex_get (kGreen1B, p-1, q+2);
                            }
                            
                            if (p >= Ncut-2 || q == 0) {
                                
                                kGreenP4B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP4B = gsl_matrix_complex_get (kGreen1B, p+2, q-1);
                            }
                            
                            OrbMixARe = GSL_REAL(kGreenP1B) + pow(2.0*cos(0.5*sqrt(3.0)*Kperp),2) * GSL_REAL(kGreenP2B)
                                        + 2.0*cos(0.5*sqrt(3.0)*Kperp) * (GSL_REAL(kGreenP3B) + GSL_REAL(kGreenP4B));
                            
                            OrbMixAIm = GSL_IMAG(kGreenP1B) + pow(2.0*cos(0.5*sqrt(3.0)*Kperp),2) * GSL_IMAG(kGreenP2B)
                                        + 2.0*cos(0.5*sqrt(3.0)*Kperp) * (GSL_IMAG(kGreenP3B) + GSL_IMAG(kGreenP4B));
                            
                            if (p == q) {
                                
                                GInvARe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkDn + Gap - 2.0*Lsoc * sin(sqrt(3.0)*Kperp) - OrbMixARe;
                                GInvAIm = 0.5*Gamma - OrbMixAIm;
                                
                            } else if (p == q + 3 || p == q - 3) {
                                
                                GInvARe = 2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp) - OrbMixARe;
                                GInvAIm = - OrbMixAIm;
                                
                            } else {
                                
                                GInvARe = - OrbMixARe;
                                GInvAIm = - OrbMixAIm;
                            }
                            
                            gsl_matrix_complex_set (kGreenInv2A, p, q, gsl_complex_rect (GInvARe, GInvAIm));
                        }
                        
                        // Mod 2 : Zigzag
                        
                        if (Mod == 2) {
                            
                            if (p == 0 || q == 0) {
                                
                                kGreenP1B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP1B = gsl_matrix_complex_get (kGreen1B, p-1, q-1);
                            }
                            
                            kGreenP2B = gsl_matrix_complex_get (kGreen1B, p, q);
                            
                            if (p == Ncut-1 || q == Ncut-1) {
                                
                                kGreenP3B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP3B = gsl_matrix_complex_get (kGreen1B, p+1, q+1);
                            }
                            
                            if (p == 0 || q == Ncut-1) {
                                
                                kGreenP4B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP4B = gsl_matrix_complex_get (kGreen1B, p-1, q+1);
                            }
                            
                            if (p == Ncut-1 || q == 0) {
                                
                                kGreenP5B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP5B = gsl_matrix_complex_get (kGreen1B, p+1, q-1);
                            }
                            
                            if (q == 0) {
                                
                                kGreenP6B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP6B = gsl_matrix_complex_get (kGreen1B, p, q-1);
                            }
                            
                            if (q == Ncut-1) {
                                
                                kGreenP7B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP7B = gsl_matrix_complex_get (kGreen1B, p, q+1);
                            }
                            
                            if (p == 0) {
                                
                                kGreenP8B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP8B = gsl_matrix_complex_get (kGreen1B, p-1, q);
                            }
                            
                            if (p == Ncut-1) {
                                
                                kGreenP9B = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP9B = gsl_matrix_complex_get (kGreen1B, p+1, q);
                            }
                            
                            OrbMixARe = GSL_REAL(kGreenP1B) + GSL_REAL(kGreenP2B) + GSL_REAL(kGreenP3B) + GSL_REAL(kGreenP4B) + GSL_REAL(kGreenP5B)
                                        + cos(1.5*Kperp) * (GSL_REAL(kGreenP6B) + GSL_REAL(kGreenP7B)) - sin(1.5*Kperp) * (GSL_IMAG(kGreenP6B) + GSL_IMAG(kGreenP7B))
                                        + cos(1.5*Kperp) * (GSL_REAL(kGreenP8B) + GSL_REAL(kGreenP9B)) + sin(1.5*Kperp) * (GSL_IMAG(kGreenP8B) + GSL_IMAG(kGreenP9B));
                            
                            OrbMixAIm = GSL_IMAG(kGreenP1B) + GSL_IMAG(kGreenP2B) + GSL_IMAG(kGreenP3B) + GSL_IMAG(kGreenP4B) + GSL_IMAG(kGreenP5B)
                                        + cos(1.5*Kperp) * (GSL_IMAG(kGreenP6B) + GSL_IMAG(kGreenP7B)) + sin(1.5*Kperp) * (GSL_REAL(kGreenP6B) + GSL_REAL(kGreenP7B))
                                        + cos(1.5*Kperp) * (GSL_IMAG(kGreenP8B) + GSL_IMAG(kGreenP9B)) - sin(1.5*Kperp) * (GSL_REAL(kGreenP8B) + GSL_REAL(kGreenP9B));
                            
                            if (p == q) {
                                
                                GInvARe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkDn + Gap - OrbMixARe;
                                GInvAIm = 0.5*Gamma - OrbMixAIm;
                                
                            } else if (p == q + 1) {
                                
                                GInvARe = - OrbMixARe;
                                GInvAIm = - 2.0*Lsoc * cos(1.5*Kperp) - OrbMixAIm;
                                
                            } else if (p == q - 1) {
                                
                                GInvARe = - OrbMixARe;
                                GInvAIm = 2.0*Lsoc * cos(1.5*Kperp) - OrbMixAIm;
                                
                            } else if (p == q + 2) {
                                
                                GInvARe = - OrbMixARe;
                                GInvAIm = Lsoc - OrbMixAIm;
                                
                            } else if (p == q - 2) {
                                
                                GInvARe = - OrbMixARe;
                                GInvAIm = - Lsoc - OrbMixAIm;
                                
                            } else {
                                
                                GInvARe = - OrbMixARe;
                                GInvAIm = - OrbMixAIm;
                            }
                            
                            gsl_matrix_complex_set (kGreenInv2A, p, q, gsl_complex_rect (GInvARe, GInvAIm));
                        }
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
                
                gsl_complex kGreenA = gsl_matrix_complex_get (kGreen2A, n, n);
                
                // Semi-local Dos
                
                SemiLocDosDnASub[iw+Fmax*(n-rank)/Ncpu] = - GSL_IMAG(kGreenA) / M_PI;
                
                // Free the previous allocation
                
                gsl_permutation_free (perm2A);
                gsl_matrix_complex_free (kGreen2A);
                gsl_matrix_complex_free (kGreenInv2A);
                
            } else {
                
                // Semi-local Dos
                
                SemiLocDosDnASub[iw+Fmax*(n-rank)/Ncpu] = 0;
            }
        }
    }
    
    return SemiLocDosDnASub;
}


/************************************************************************************************/
/***************************** [Subroutine] Semi-Local Dos (Up, B) ******************************/
/************************************************************************************************/

double *SubSemiLocDosUpB (int rank, double EkUp, double Kperp)
{
    // Definition of local variables
    
    int iw, n, p, q, s;
    double FloHamRe, FloHamIm;
    double OrbMixBRe, OrbMixBIm, GInvBRe, GInvBIm;
    static double SemiLocDosUpBSub[Icut/Ncpu];
    
    // Find the semi-local Dos at low field
    
    for (iw = Fmax*rank/Ncpu; iw <= Fmax*(rank+1)/Ncpu-1; iw++)
    {
        for (n = 0; n <= Ncut-1; n++)
        {
            if (n >= (Ncut-1)/2 - 6 && n <= (Ncut-1)/2 + 6) {
                
                // 1-band retarded lattice Green's function
                
                gsl_matrix_complex *kGreenInv1A = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        // Mod 1 : Armchair
                        
                        if (Mod == 1) {
                            
                            if (p == q) {
                                
                                FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkUp + Gap + 2.0*Lsoc * sin(sqrt(3.0)*Kperp);
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
                                
                            } else if (p == q + 3 || p == q - 3) {
                                
                                FloHamRe = - 2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp);
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0));
                                
                            } else {
                                
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, 0));
                            }
                        }
                        
                        // Mod 2 : Zigzag
                        
                        if (Mod == 2) {
                            
                            if (p == q) {
                                
                                FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkUp + Gap;
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
                                
                            } else if (p == q + 1) {
                                
                                FloHamIm = 2.0*Lsoc * cos(1.5*Kperp);
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, FloHamIm));
                                
                            } else if (p == q - 1) {
                                
                                FloHamIm = -2.0*Lsoc * cos(1.5*Kperp);
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, FloHamIm));
                                
                            } else if (p == q + 2) {
                                
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, -Lsoc));
                                
                            } else if (p == q - 2) {
                                
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, Lsoc));
                                
                            } else {
                                
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, 0));
                            }
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
                        // Mod 1 : Armchair
                        
                        if (Mod == 1) {
                            
                            if (p <= 1 || q <= 1) {
                                
                                kGreenP1A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP1A = gsl_matrix_complex_get (kGreen1A, p-2, q-2);
                            }
                            
                            if (p == Ncut-1 || q == Ncut-1) {
                                
                                kGreenP2A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP2A = gsl_matrix_complex_get (kGreen1A, p+1, q+1);
                            }
                            
                            if (p == Ncut-1 || q <= 1) {
                                
                                kGreenP3A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP3A = gsl_matrix_complex_get (kGreen1A, p+1, q-2);
                            }
                            
                            if (p <= 1 || q == Ncut-1) {
                                
                                kGreenP4A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP4A = gsl_matrix_complex_get (kGreen1A, p-2, q+1);
                            }
                            
                            OrbMixBRe = GSL_REAL(kGreenP1A) + pow(2.0*cos(0.5*sqrt(3.0)*Kperp),2) * GSL_REAL(kGreenP2A)
                            + 2.0*cos(0.5*sqrt(3.0)*Kperp) * (GSL_REAL(kGreenP3A) + GSL_REAL(kGreenP4A));
                            
                            OrbMixBIm = GSL_IMAG(kGreenP1A) + pow(2.0*cos(0.5*sqrt(3.0)*Kperp),2) * GSL_IMAG(kGreenP2A)
                            + 2.0*cos(0.5*sqrt(3.0)*Kperp) * (GSL_IMAG(kGreenP3A) + GSL_IMAG(kGreenP4A));
                            
                            if (p == q) {
                                
                                GInvBRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkUp - Gap - 2.0*Lsoc * sin(sqrt(3.0)*Kperp) - OrbMixBRe;
                                GInvBIm = 0.5*Gamma - OrbMixBIm;
                                
                            } else if (p == q + 3 || p == q - 3) {
                                
                                GInvBRe = 2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp) - OrbMixBRe;
                                GInvBIm = - OrbMixBIm;
                                
                            } else {
                                
                                GInvBRe = - OrbMixBRe;
                                GInvBIm = - OrbMixBIm;
                            }
                            
                            gsl_matrix_complex_set (kGreenInv2B, p, q, gsl_complex_rect (GInvBRe, GInvBIm));
                        }
                        
                        // Mod 2 : Zigzag
                        
                        if (Mod == 2) {
                            
                            if (p == 0 || q == 0) {
                                
                                kGreenP1A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP1A = gsl_matrix_complex_get (kGreen1A, p-1, q-1);
                            }
                            
                            kGreenP2A = gsl_matrix_complex_get (kGreen1A, p, q);
                            
                            if (p == Ncut-1 || q == Ncut-1) {
                                
                                kGreenP3A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP3A = gsl_matrix_complex_get (kGreen1A, p+1, q+1);
                            }
                            
                            if (p == 0 || q == Ncut-1) {
                                
                                kGreenP4A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP4A = gsl_matrix_complex_get (kGreen1A, p-1, q+1);
                            }
                            
                            if (p == Ncut-1 || q == 0) {
                                
                                kGreenP5A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP5A = gsl_matrix_complex_get (kGreen1A, p+1, q-1);
                            }
                            
                            if (q == 0) {
                                
                                kGreenP6A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP6A = gsl_matrix_complex_get (kGreen1A, p, q-1);
                            }
                            
                            if (q == Ncut-1) {
                                
                                kGreenP7A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP7A = gsl_matrix_complex_get (kGreen1A, p, q+1);
                            }
                            
                            if (p == 0) {
                                
                                kGreenP8A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP8A = gsl_matrix_complex_get (kGreen1A, p-1, q);
                            }
                            
                            if (p == Ncut-1) {
                                
                                kGreenP9A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP9A = gsl_matrix_complex_get (kGreen1A, p+1, q);
                            }
                            
                            OrbMixBRe = GSL_REAL(kGreenP1A) + GSL_REAL(kGreenP2A) + GSL_REAL(kGreenP3A) + GSL_REAL(kGreenP4A) + GSL_REAL(kGreenP5A)
                            + cos(1.5*Kperp) * (GSL_REAL(kGreenP6A) + GSL_REAL(kGreenP7A)) + sin(1.5*Kperp) * (GSL_IMAG(kGreenP6A) + GSL_IMAG(kGreenP7A))
                            + cos(1.5*Kperp) * (GSL_REAL(kGreenP8A) + GSL_REAL(kGreenP9A)) - sin(1.5*Kperp) * (GSL_IMAG(kGreenP8A) + GSL_IMAG(kGreenP9A));
                            
                            OrbMixBIm = GSL_IMAG(kGreenP1A) + GSL_IMAG(kGreenP2A) + GSL_IMAG(kGreenP3A) + GSL_IMAG(kGreenP4A) + GSL_IMAG(kGreenP5A)
                            + cos(1.5*Kperp) * (GSL_IMAG(kGreenP6A) + GSL_IMAG(kGreenP7A)) - sin(1.5*Kperp) * (GSL_REAL(kGreenP6A) + GSL_REAL(kGreenP7A))
                            + cos(1.5*Kperp) * (GSL_IMAG(kGreenP8A) + GSL_IMAG(kGreenP9A)) + sin(1.5*Kperp) * (GSL_REAL(kGreenP8A) + GSL_REAL(kGreenP9A));
                            
                            if (p == q) {
                                
                                GInvBRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkUp - Gap - OrbMixBRe;
                                GInvBIm = 0.5*Gamma - OrbMixBIm;
                                
                            } else if (p == q + 1) {
                                
                                GInvBRe = - OrbMixBRe;
                                GInvBIm = -2.0*Lsoc * cos(1.5*Kperp) - OrbMixBIm;
                                
                            } else if (p == q - 1) {
                                
                                GInvBRe = - OrbMixBRe;
                                GInvBIm = 2.0*Lsoc * cos(1.5*Kperp) - OrbMixBIm;
                                
                            } else if (p == q + 2) {
                                
                                GInvBRe = - OrbMixBRe;
                                GInvBIm = Lsoc - OrbMixBIm;
                                
                            } else if (p == q - 2) {
                                
                                GInvBRe = - OrbMixBRe;
                                GInvBIm = - Lsoc - OrbMixBIm;
                                
                            } else {
                                
                                GInvBRe = - OrbMixBRe;
                                GInvBIm = - OrbMixBIm;
                            }
                            
                            gsl_matrix_complex_set (kGreenInv2B, p, q, gsl_complex_rect (GInvBRe, GInvBIm));
                        }
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
                
                gsl_complex kGreenB = gsl_matrix_complex_get (kGreen2B, n, n);
                
                // Semi-local Dos
                
                SemiLocDosUpBSub[iw+Fmax*(n-rank)/Ncpu] = - GSL_IMAG(kGreenB) / M_PI;
                
                // Free the previous allocation
                
                gsl_permutation_free (perm2B);
                gsl_matrix_complex_free (kGreen2B);
                gsl_matrix_complex_free (kGreenInv2B);
                
            } else {
                
                // Semi-local Dos
                
                SemiLocDosUpBSub[iw+Fmax*(n-rank)/Ncpu] = 0;
            }
        }
    }
    
    return SemiLocDosUpBSub;
}


/************************************************************************************************/
/***************************** [Subroutine] Semi-Local Dos (Dn, B) ******************************/
/************************************************************************************************/

double *SubSemiLocDosDnB (int rank, double EkDn, double Kperp)
{
    // Definition of local variables
    
    int iw, n, p, q, s;
    double FloHamRe, FloHamIm;
    double OrbMixBRe, OrbMixBIm, GInvBRe, GInvBIm;
    static double SemiLocDosDnBSub[Icut/Ncpu];
    
    // Find the semi-local Dos at low field
    
    for (iw = Fmax*rank/Ncpu; iw <= Fmax*(rank+1)/Ncpu-1; iw++)
    {
        for (n = 0; n <= Ncut-1; n++)
        {
            if (n >= (Ncut-1)/2 - 6 && n <= (Ncut-1)/2 + 6) {
                
                // 1-band retarded lattice Green's function
                
                gsl_matrix_complex *kGreenInv1A = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        // Mod 1 : Armchair
                        
                        if (Mod == 1) {
                            
                            if (p == q) {
                                
                                FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkDn + Gap - 2.0*Lsoc * sin(sqrt(3.0)*Kperp);
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
                                
                            } else if (p == q + 3 || p == q - 3) {
                                
                                FloHamRe = 2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp);
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0));
                                
                            } else {
                                
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, 0));
                            }
                        }
                        
                        // Mod 2 : Zigzag
                        
                        if (Mod == 2) {
                            
                            if (p == q) {
                                
                                FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkDn + Gap;
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
                                
                            } else if (p == q + 1) {
                                
                                FloHamIm = - 2.0*Lsoc * cos(1.5*Kperp);
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, FloHamIm));
                                
                            } else if (p == q - 1) {
                                
                                FloHamIm = 2.0*Lsoc * cos(1.5*Kperp);
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, FloHamIm));
                                
                            } else if (p == q + 2) {
                                
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, Lsoc));
                                
                            } else if (p == q - 2) {
                                
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, - Lsoc));
                                
                            } else {
                                
                                gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, 0));
                            }
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
                        // Mod 1 : Armchair
                        
                        if (Mod == 1) {
                            
                            if (p <= 1 || q <= 1) {
                                
                                kGreenP1A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP1A = gsl_matrix_complex_get (kGreen1A, p-2, q-2);
                            }
                            
                            if (p == Ncut-1 || q == Ncut-1) {
                                
                                kGreenP2A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP2A = gsl_matrix_complex_get (kGreen1A, p+1, q+1);
                            }
                            
                            if (p == Ncut-1 || q <= 1) {
                                
                                kGreenP3A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP3A = gsl_matrix_complex_get (kGreen1A, p+1, q-2);
                            }
                            
                            if (p <= 1 || q == Ncut-1) {
                                
                                kGreenP4A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP4A = gsl_matrix_complex_get (kGreen1A, p-2, q+1);
                            }
                            
                            OrbMixBRe = GSL_REAL(kGreenP1A) + pow(2.0*cos(0.5*sqrt(3.0)*Kperp),2) * GSL_REAL(kGreenP2A)
                                        + 2.0*cos(0.5*sqrt(3.0)*Kperp) * (GSL_REAL(kGreenP3A) + GSL_REAL(kGreenP4A));
                            
                            OrbMixBIm = GSL_IMAG(kGreenP1A) + pow(2.0*cos(0.5*sqrt(3.0)*Kperp),2) * GSL_IMAG(kGreenP2A)
                                        + 2.0*cos(0.5*sqrt(3.0)*Kperp) * (GSL_IMAG(kGreenP3A) + GSL_IMAG(kGreenP4A));
                            
                            if (p == q) {
                                
                                GInvBRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkDn - Gap + 2.0*Lsoc * sin(sqrt(3.0)*Kperp) - OrbMixBRe;
                                GInvBIm = 0.5*Gamma - OrbMixBIm;
                                
                            } else if (p == q + 3 || p == q - 3) {
                                
                                GInvBRe = - 2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp) - OrbMixBRe;
                                GInvBIm = - OrbMixBIm;
                                
                            } else {
                                
                                GInvBRe = - OrbMixBRe;
                                GInvBIm = - OrbMixBIm;
                            }
                            
                            gsl_matrix_complex_set (kGreenInv2B, p, q, gsl_complex_rect (GInvBRe, GInvBIm));
                        }
                        
                        // Mod 2 : Zigzag
                        
                        if (Mod == 2) {
                            
                            if (p == 0 || q == 0) {
                                
                                kGreenP1A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP1A = gsl_matrix_complex_get (kGreen1A, p-1, q-1);
                            }
                            
                            kGreenP2A = gsl_matrix_complex_get (kGreen1A, p, q);
                            
                            if (p == Ncut-1 || q == Ncut-1) {
                                
                                kGreenP3A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP3A = gsl_matrix_complex_get (kGreen1A, p+1, q+1);
                            }
                            
                            if (p == 0 || q == Ncut-1) {
                                
                                kGreenP4A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP4A = gsl_matrix_complex_get (kGreen1A, p-1, q+1);
                            }
                            
                            if (p == Ncut-1 || q == 0) {
                                
                                kGreenP5A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP5A = gsl_matrix_complex_get (kGreen1A, p+1, q-1);
                            }
                            
                            if (q == 0) {
                                
                                kGreenP6A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP6A = gsl_matrix_complex_get (kGreen1A, p, q-1);
                            }
                            
                            if (q == Ncut-1) {
                                
                                kGreenP7A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP7A = gsl_matrix_complex_get (kGreen1A, p, q+1);
                            }
                            
                            if (p == 0) {
                                
                                kGreenP8A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP8A = gsl_matrix_complex_get (kGreen1A, p-1, q);
                            }
                            
                            if (p == Ncut-1) {
                                
                                kGreenP9A = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                kGreenP9A = gsl_matrix_complex_get (kGreen1A, p+1, q);
                            }
                            
                            OrbMixBRe = GSL_REAL(kGreenP1A) + GSL_REAL(kGreenP2A) + GSL_REAL(kGreenP3A) + GSL_REAL(kGreenP4A) + GSL_REAL(kGreenP5A)
                                        + cos(1.5*Kperp) * (GSL_REAL(kGreenP6A) + GSL_REAL(kGreenP7A)) + sin(1.5*Kperp) * (GSL_IMAG(kGreenP6A) + GSL_IMAG(kGreenP7A))
                                        + cos(1.5*Kperp) * (GSL_REAL(kGreenP8A) + GSL_REAL(kGreenP9A)) - sin(1.5*Kperp) * (GSL_IMAG(kGreenP8A) + GSL_IMAG(kGreenP9A));
                            
                            OrbMixBIm = GSL_IMAG(kGreenP1A) + GSL_IMAG(kGreenP2A) + GSL_IMAG(kGreenP3A) + GSL_IMAG(kGreenP4A) + GSL_IMAG(kGreenP5A)
                                        + cos(1.5*Kperp) * (GSL_IMAG(kGreenP6A) + GSL_IMAG(kGreenP7A)) - sin(1.5*Kperp) * (GSL_REAL(kGreenP6A) + GSL_REAL(kGreenP7A))
                                        + cos(1.5*Kperp) * (GSL_IMAG(kGreenP8A) + GSL_IMAG(kGreenP9A)) + sin(1.5*Kperp) * (GSL_REAL(kGreenP8A) + GSL_REAL(kGreenP9A));
                            
                            if (p == q) {
                                
                                GInvBRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + EkDn - Gap - OrbMixBRe;
                                GInvBIm = 0.5*Gamma - OrbMixBIm;
                                
                            } else if (p == q + 1) {
                                
                                GInvBRe = - OrbMixBRe;
                                GInvBIm = 2.0*Lsoc * cos(1.5*Kperp) - OrbMixBIm;
                                
                            } else if (p == q - 1) {
                                
                                GInvBRe = - OrbMixBRe;
                                GInvBIm = - 2.0*Lsoc * cos(1.5*Kperp) - OrbMixBIm;
                                
                            } else if (p == q + 2) {
                                
                                GInvBRe = - OrbMixBRe;
                                GInvBIm = - Lsoc - OrbMixBIm;
                                
                            } else if (p == q - 2) {
                                
                                GInvBRe = - OrbMixBRe;
                                GInvBIm = Lsoc - OrbMixBIm;
                                
                            } else {
                                
                                GInvBRe = - OrbMixBRe;
                                GInvBIm = - OrbMixBIm;
                            }
                            
                            gsl_matrix_complex_set (kGreenInv2B, p, q, gsl_complex_rect (GInvBRe, GInvBIm));
                        }
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
                
                gsl_complex kGreenB = gsl_matrix_complex_get (kGreen2B, n, n);
                
                // Semi-local Dos
                
                SemiLocDosDnBSub[iw+Fmax*(n-rank)/Ncpu] = - GSL_IMAG(kGreenB) / M_PI;
                
                // Free the previous allocation
                
                gsl_permutation_free (perm2B);
                gsl_matrix_complex_free (kGreen2B);
                gsl_matrix_complex_free (kGreenInv2B);
                
            } else {
                
                // Semi-local Dos
                
                SemiLocDosDnBSub[iw+Fmax*(n-rank)/Ncpu] = 0;
            }
        }
    }
    
    return SemiLocDosDnBSub;
}
