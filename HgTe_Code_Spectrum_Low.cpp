/************************************************************************************************/
/************* [System] 2D Bloch oscillator (HgTe heterostructure)				  ***************/
/************* [Method] Floquet-Keldysh Green's function						  ***************/
/************* [Programmer] Dr. Woo-Ram Lee (wrlee@kias.re.kr)					  ***************/
/************************************************************************************************/

// MPI routines for parallel computing

#include <mpi.h>

// Standard libraries

#include <stdio.h>
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

int const		Mod		= 2;				// (1) Equilibrium; (2) Finite Field

double const	ScaFac	= 1.0;				// scaling factor for the side direction

int	const		Ncpu	= 5;				// total number of CPUs, which are really used for calculation

int const		Iband	= Ncpu * 3 * 45;	// grid number of frequency domain in a bandwidth
int const		Imax	= Ncpu * 3 * 85*29;	// maximum grid number of frequency domain in a full range
double const	Domega	= ScaFac/Iband;		// grid in frequency domain

int const		Kband	= Ncpu * 3 * 35;	// grid number of k_perp within 1 B.W.
int const		Kcut	= Kband * 7;		// grid number of k_perp
double const	Komega	= 1.0/Kband;		// grid in k_perp domain

// Physical constants

double const	Temp	= 0.000001*Domega;	// Temperature (Zero Temp = 0.000001*Domega)
double const	Gamma	= 0.01;				// System-bath scattering rate
double const	A		= 0.6;				// off-diagonal dispersion (+, in unit 4D)
double const	B		= 0.6;				// spin-orbit coupling strength (+, in unit 4D)
double const	M		= -0.3;				// mass gap (+: trivial, -: nontrivial, in unit 4D)
double const	Kperp	= 0.0 * M_PI;		// Transverse momentum: - M_PI ~ M_PI

// Define subroutine

double *SubLocDos (int rank, int Fmax, int Ncut, int Icut, double Field, double Bloch);


/************************************************************************************************/
/************************************** [Main routine] ******************************************/
/************************************************************************************************/

main(int argc, char **argv)
{	
	// Open the saving file
	
	FILE *f1;
	f1 = fopen("HgTeDosA6B6Mm3K0Low3","wt");
	
	// Initiate the MPI system
	
	int rank;		// Rank of my CPU
	int source;		// CPU sending message
	int dest = 0;	// CPU receiving message
	int tag = 0;	// Indicator distinguishing messages
	
	MPI_Status status;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	
	// Declare variables
	
	int iw, n, i, j, Fmax, Ncut, Icut;
	double Field, Bloch;
	
	// Loop for electric field
	
	for (i = 3; i <= 3; i += 2)        // Equil (39), Low Field (3), (5), (7), (9,19), (21,39)
	{
		// Define local variables
		
		Fmax = Ncpu * i;				// grid number of Bloch frequency
		
		for (j = 1; j <= 999; j += 2)
		{
			if (1.0*j >= 20.0*Iband/ScaFac/Fmax) {
				
				Ncut = j;				// Cutoff of Floquet mode
				break;
			}
		}
		
		Icut  = Ncut*Fmax;				// grid number of frequency domain in a full range
		Field = Fmax*1.0/Iband;			// Electric field
		Bloch = Fmax*Domega;			// Bloch frequency
		
		// Allocate functions
		
		double *LocDosSub;
		double LocDos[Icut];
		
		// Local Dos (E1 + H1)
		
		if (rank >= 1 && rank <= Ncpu-1) {
			
			LocDosSub = SubLocDos (rank, Fmax, Ncut, Icut, Field, Bloch);
			MPI_Send (LocDosSub, Imax/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
			
		} else if (rank == 0) {
			
			for (source = 0; source <= Ncpu-1; source++)
			{
				if (source == 0) {
					
					LocDosSub = SubLocDos (source, Fmax, Ncut, Icut, Field, Bloch);
					
				} else {
					
					MPI_Recv (LocDosSub, Imax/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
				}
				
				for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++) 
				{
					for (n = 0; n <= Ncut-1; n++)
					{
						LocDos[iw+n*Fmax] = LocDosSub[iw+Fmax*(n-source)/Ncpu];
					}
				}
			}
		}
		
		// Save data on output files
		
		if (rank == 0) {
			
			if (Mod == 1) {
				
				Field = 0;
			}
			
			printf("The calculation for Field = %f was completed.\n", Field);
			
			for (iw = 0; iw <= Icut-1; iw++)
			{				
				fprintf(f1, "%f %f %e \n", Field, (iw-(Icut-1.0)/2.0)*Domega, LocDos[iw]);
			}
			
			fprintf(f1, "\n\n");
		}
	}
		
	// Close save file
	
	fclose(f1);
	
	// Finish MPI system
	
	MPI_Finalize();
}


/************************************************************************************************/
/**************************** [Subroutine] Local Green's functions ******************************/
/************************************************************************************************/

double *SubLocDos (int rank, int Fmax, int Ncut, int Icut, double Field, double Bloch)
{
	// Definition of local variables
	
	int ik, iw, n, p, q, s;
	double d1, d2, d3, ek, Ek1, Ek2, ReG1, ImG1, ReG2, ImG2;
	double FloHamRe, FloHamIm, FloHamDRe, FloHamDIm;
	double OrbMixARe, OrbMixAIm, GInvARe, GInvAIm;
	double OrbMixBRe, OrbMixBIm, GInvBRe, GInvBIm;
	static double LocDosSub[Imax/Ncpu];
	
	for (iw = Fmax*rank/Ncpu; iw <= Fmax*(rank+1)/Ncpu-1; iw++)
	{
		for (n = 0; n <= Ncut-1; n++)
		{
			LocDosSub[iw+Fmax*(n-rank)/Ncpu] = 0;
			
			// Mod 1 : Side direction - Equilibrium
			
			if (Mod == 1) {
				
				for (ik = 0; ik <= Kcut-1; ik++)
				{
					if ( fabs((ik-(Kcut-1.0)/2.0)*Komega) <= M_PI ) {
						
						// Dispersion relation
						
						d1 = A * sin((ik-(Kcut-1.0)/2.0)*Komega);
						d2 = - A * sin(Kperp);
						d3 = M + 2.0 * B * (2.0 - cos((ik-(Kcut-1.0)/2.0)*Komega) - cos(Kperp));
						ek = 0.5 * (2.0 - cos((ik-(Kcut-1.0)/2.0)*Komega) - cos(Kperp));
						
						Ek1 = ek - sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2));
						Ek2 = ek + sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2));
						
						// Local Green's function
						
						ReG1 = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0) * Bloch - Ek1;
						ReG2 = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0) * Bloch - Ek2;
						
						// Semi-local Dos
						
						LocDosSub[iw+Fmax*(n-rank)/Ncpu] += Komega/(2.0*M_PI) * 1.0/M_PI
                            * (0.5*Gamma / (pow(ReG1,2) + pow(0.5*Gamma,2))
                            + 0.5*Gamma / (pow(ReG2,2) + pow(0.5*Gamma,2)));
					}
				}
			}
			
			
			// Mod 2 : Side direction - Finite Field
			
			if (Mod == 2) {
				
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
							
							GInvARe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch  
									  - M - 2.0 * (0.25 + B) * (2.0 - cos(Kperp)) - OrbMixARe;
							GInvAIm = 0.5*Gamma - OrbMixAIm;
							
							GInvBRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch 
									  + M - 2.0 * (0.25 - B) * (2.0 - cos(Kperp)) - OrbMixBRe;
							GInvBIm = 0.5*Gamma - OrbMixBIm;
							
						} else if (p == q + 1) {
							
							GInvARe = 0.25 + B - OrbMixARe;
							GInvAIm = - OrbMixAIm;
							
							GInvBRe = 0.25 - B - OrbMixBRe;
							GInvBIm = - OrbMixBIm;
							
						} else if (p == q - 1) {
							
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
				
				// 2-band retarded local Green's function (A: E1 state, B: H1 state)
				
                gsl_complex kGreenA = gsl_matrix_complex_get (kGreen2A, n, n);
				gsl_complex kGreenB = gsl_matrix_complex_get (kGreen2B, n, n);
				
				// Semi-local Dos (E1 + H1)
				
				LocDosSub[iw+Fmax*(n-rank)/Ncpu] = - (GSL_IMAG(kGreenA) + GSL_IMAG(kGreenB)) / M_PI;
				
				// Free the previous allocation
				
				gsl_permutation_free (perm2A);
                gsl_matrix_complex_free (kGreen2A);
                gsl_matrix_complex_free (kGreenInv2A);
				
                gsl_permutation_free (perm2B);
                gsl_matrix_complex_free (kGreen2B);
                gsl_matrix_complex_free (kGreenInv2B);
			}
		}
	}
	
	return LocDosSub;
}
