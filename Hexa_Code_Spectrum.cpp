/************************************************************************************************/
/************* [System] 2D Bloch oscillator (Hexagonal lattice)					  ***************/
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

int const		Mod    = 2;					// (1) Armchair - Equilibrium
											// (2) Armchair - Finite Field
											// (3) Zigzag   - Equilibrium
											// (4) Zigzag   - Finite Field

double const	ScaFac = 0.5;				// Scaling factor
											// Equilibrium (1),(3) 1.0
											// Armchair    (2) 0.5
											// Zigzag      (4) 0.5*sqrt(3.0)

int	const		Ncpu   = 15;				// total number of CPUs, which are really used for calculation
int const		Iband  = Ncpu * 45;			// grid number of frequency domain in a bandwidth
int const		Imax   = Ncpu * 85*29;		// maximum grid number of frequency domain in a full range
double const	Domega = ScaFac/Iband;		// grid in frequency domain
int const		Kband  = Ncpu * 35;			// grid number of k_perp within 1 B.W.
int const		Kcut   = Kband * 7;			// grid number of k_perp
double const	Komega = 1.0/Kband;			// grid in k_perp domain

// Physical constants

double const	Temp   = 0.000001*Domega;	// Temperature (Zero Temp = 0.000001*Domega)
double const	Gamma  = 0.01;				// System-bath scattering rate
double const	Gap    = 0.0;				// Mass gap
double const	Lsoc   = 0.0;				// Spin-orbit coupling
double const	Kperp  = -0.75*M_PI/sqrt(3.0);		// Transverse momentum 
											// (2),(3) -M_PI/sqrt(3.0), M_PI/sqrt(3.0)
											// (5),(6) -M_PI/3.0, M_PI/3.0

// Define subroutine

double *SubLocGreen (int rank, int Fmax, int Ncut, int Icut, double Field, double Bloch);


/************************************************************************************************/
/************************************** [Main routine] ******************************************/
/************************************************************************************************/

main(int argc, char **argv)
{	
	// Open the saving file
	
	FILE *f1;
	f1 = fopen("HexaAcG000L000KM075","wt");
	
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
	
	for (i = 3; i <= 181; i += 2)	// Equil 45, Low Field (3,21), High Field (23,181)
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
		
		double *LocGreenSub;
		double RetLocGreen[2*Icut];
		double LocDos[Icut];
		
		// Local Green's function
		
		if (rank >= 1 && rank <= Ncpu-1) {
			
			LocGreenSub = SubLocGreen (rank, Fmax, Ncut, Icut, Field, Bloch);
			MPI_Send (LocGreenSub, 2*Imax/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
			
		} else if (rank == 0) {
			
			for (source = 0; source <= Ncpu-1; source++)
			{
				if (source == 0) {
					
					LocGreenSub = SubLocGreen (source, Fmax, Ncut, Icut, Field, Bloch);
					
				} else {
					
					MPI_Recv (LocGreenSub, 2*Imax/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
				}
				
				for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++) 
				{
					for (n = 0; n <= Ncut-1; n++)
					{
						RE1(RetLocGreen,iw+n*Fmax) = RE1(LocGreenSub,iw+Fmax*(n-source)/Ncpu);
						IM1(RetLocGreen,iw+n*Fmax) = IM1(LocGreenSub,iw+Fmax*(n-source)/Ncpu);
					}
				}
			}
		}
		
		// Local energy spectrum
		
		if (rank == 0) {
			
			for (iw = 0; iw <= Icut-1; iw++)
			{	
				LocDos[iw] = -IM1(RetLocGreen,iw)/M_PI;
			}
		}
		
		// Save data on output files
		
		if (rank == 0) {
			
			printf("The calculation for Field = %f was completed.\n", Field);
			
			if (Mod == 1 || Mod == 4) {
				
				for (iw = 0; iw <= Icut-1; iw++)
				{				
					fprintf(f1, "%f %f %e \n", 0.0, (iw-(Icut-1.0)/2.0)*Domega, LocDos[iw]);
				}
				
			} else {
				
				for (iw = 0; iw <= Icut-1; iw++)
				{				
					fprintf(f1, "%f %f %e \n", Field, (iw-(Icut-1.0)/2.0)*Domega, LocDos[iw]);
				}
			}
			
			fprintf(f1, "\n");
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

double *SubLocGreen (int rank, int Fmax, int Ncut, int Icut, double Field, double Bloch)
{
	// Definition of local variables
	
	int ik, iw, n, p, q, s;
	double FloHamRe, FloHamIm, FloHamDRe, FloHamDIm, Ek, EkAA, Ek1, Ek2, ReG1, ReG2, ImG2, FermiDist;
	double OrbMixARe, OrbMixAIm, GInvARe, GInvAIm;
	double OrbMixBRe, OrbMixBIm, GInvBRe, GInvBIm;
	static double LocGreenSub[2*Imax/Ncpu];
	
	for (iw = Fmax*rank/Ncpu; iw <= Fmax*(rank+1)/Ncpu-1; iw++)
	{
		for (n = 0; n <= Ncut-1; n++)
		{
			RE1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) = 0;
			IM1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) = 0;
			
			
			// Mod 1 : Armchair - Equilibrium
			
			if (Mod == 1) {
				
				for (ik = 0; ik <= Kcut-1; ik++)
				{
					if ( (ik-(Kcut-1.0)/2.0)*Komega >= 0 && (ik-(Kcut-1.0)/2.0)*Komega <= 4.0*M_PI/3.0 ) {
						
						// Dispersion relation
						
						EkAA = Lsoc * (4.0*cos(1.5*(ik-(Kcut-1.0)/2.0)*Komega)*sin(0.5*sqrt(3.0)*Kperp) - 2.0*sin(sqrt(3.0)*Kperp));
						Ek = 4.0*cos(1.5*(ik-(Kcut-1.0)/2.0)*Komega)*cos(0.5*sqrt(3.0)*Kperp) + 2.0*cos(sqrt(3.0)*Kperp) + 3.0;
						
						// Local Green's function
						
						ReG1 = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0) * Bloch - 0.5*Gap + EkAA;
						ReG2 = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0) * Bloch + 0.5*Gap - EkAA
							- Ek * ReG1 / (pow(ReG1,2) + pow(0.5*Gamma,2));
						ImG2 = 0.5*Gamma * (1.0 + Ek / (pow(ReG1,2) + pow(0.5*Gamma,2)));
						
						FermiDist = 1.0 / (1.0+exp(((iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0) * Bloch)/Temp));
						
						RE1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
							Komega*3.0/(4.0*M_PI) * ReG2 / (pow(ReG2,2) + pow(ImG2,2));
						IM1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
							Komega*3.0/(4.0*M_PI) * (-ImG2) / (pow(ReG2,2) + pow(ImG2,2));
					}
				}
			}
			
			
			// Mod 2 : Armchair - Finite Field
			
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
								+ 0.5*Gap + 2.0*Lsoc * sin(sqrt(3.0)*Kperp);
							
							gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
							
							FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch 
								- 0.5*Gap - 2.0*Lsoc * sin(sqrt(3.0)*Kperp);
							
							gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
							
						} else if (p == q + 3) {
							
							FloHamRe = -2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp);
							
							gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0));
							gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (-FloHamRe, 0));
							
						} else if (p == q - 3) {
							
							FloHamRe = -2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp);
							
							gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0));
							gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (-FloHamRe, 0));
							
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
				gsl_matrix_complex *kGreenInv2B = gsl_matrix_complex_alloc (Ncut, Ncut);
				gsl_complex kGreenP1A, kGreenP2A, kGreenP3A;
				gsl_complex kGreenP4A, kGreenP5A, kGreenP6A;
				gsl_complex kGreenP7A, kGreenP8A, kGreenP9A;
				gsl_complex kGreenP1B, kGreenP2B, kGreenP3B;
				gsl_complex kGreenP4B, kGreenP5B, kGreenP6B;
				gsl_complex kGreenP7B, kGreenP8B, kGreenP9B;
				
				for (p = 0; p <= Ncut-1; p++)
				{
					for (q = 0; q <= Ncut-1; q++)
					{
						if (p == 0 || q == 0) {
							
							kGreenP1B = gsl_complex_rect (0, 0);
							
						} else {
							
							kGreenP1B = gsl_matrix_complex_get (kGreen1B, p-1, q-1);
						}
						
						if (p >= Ncut-2 || q >= Ncut-2) {
							
							kGreenP2B = gsl_complex_rect (0, 0);
							
						} else {
							
							kGreenP2B = gsl_matrix_complex_get (kGreen1B, p+2, q+2);
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
						
						OrbMixARe = GSL_REAL(kGreenP2B) + pow(2.0*cos(0.5*sqrt(3.0)*Kperp),2) * GSL_REAL(kGreenP1B)
							+ 2.0*cos(0.5*sqrt(3.0)*Kperp) * (GSL_REAL(kGreenP3B) + GSL_REAL(kGreenP4B));
						
						OrbMixAIm = GSL_IMAG(kGreenP2B) + pow(2.0*cos(0.5*sqrt(3.0)*Kperp),2) * GSL_IMAG(kGreenP1B)
							+ 2.0*cos(0.5*sqrt(3.0)*Kperp) * (GSL_IMAG(kGreenP3B) + GSL_IMAG(kGreenP4B));
						
						if (p == Ncut-1 || q == Ncut-1) {
							
							kGreenP1A = gsl_complex_rect (0, 0);
							
						} else {
							
							kGreenP1A = gsl_matrix_complex_get (kGreen1A, p+1, q+1);
						}
						
						if (p <= 1 || q <= 1) {
							
							kGreenP2A = gsl_complex_rect (0, 0);
							
						} else {
							
							kGreenP2A = gsl_matrix_complex_get (kGreen1A, p-2, q-2);
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
						
						OrbMixBRe = GSL_REAL(kGreenP2A) + pow(2.0*cos(0.5*sqrt(3.0)*Kperp),2) * GSL_REAL(kGreenP1A)
							+ 2.0*cos(0.5*sqrt(3.0)*Kperp) * (GSL_REAL(kGreenP3A) + GSL_REAL(kGreenP4A));
						
						OrbMixBIm = GSL_IMAG(kGreenP2A) + pow(2.0*cos(0.5*sqrt(3.0)*Kperp),2) * GSL_IMAG(kGreenP1A)
							+ 2.0*cos(0.5*sqrt(3.0)*Kperp) * (GSL_IMAG(kGreenP3A) + GSL_IMAG(kGreenP4A));
						
						if (p == q) {
							
							GInvARe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + 0.5*Gap + 2.0*Lsoc * sin(sqrt(3.0)*Kperp) - OrbMixARe;
							GInvAIm = 0.5*Gamma - OrbMixAIm;
							
							GInvBRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch - 0.5*Gap - 2.0*Lsoc * sin(sqrt(3.0)*Kperp) - OrbMixBRe;
							GInvBIm = 0.5*Gamma - OrbMixBIm;
							
						} else if (p == q + 3) {
							
							GInvARe = - 2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp) - OrbMixARe;
							GInvAIm = - OrbMixAIm;
							
							GInvBRe = 2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp) - OrbMixBRe;
							GInvBIm = - OrbMixBIm;
							
						} else if (p == q - 3) {
							
							GInvARe = - 2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp) - OrbMixARe;
							GInvAIm = - OrbMixAIm;
							
							GInvBRe = 2.0*Lsoc * sin(0.5*sqrt(3.0)*Kperp) - OrbMixBRe;
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
				
				RE1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) = GSL_REAL(kGreenA);
				IM1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) = GSL_IMAG(kGreenA);
				
				// Free the previous allocation
				
				gsl_permutation_free (perm2A);
				gsl_matrix_complex_free (kGreenInv2A);
				gsl_matrix_complex_free (kGreen2A);
				
				gsl_permutation_free (perm2B);
				gsl_matrix_complex_free (kGreenInv2B);
				gsl_matrix_complex_free (kGreen2B);
			}
			
			
			// Mod 3 : Zigzag - Equilibrium
			
			if (Mod == 3) {
				
				for (ik = 0; ik <= Kcut-1; ik++)
				{
					if ( (ik-(Kcut-1.0)/2.0)*Komega >= -M_PI/sqrt(3.0) && (ik-(Kcut-1.0)/2.0)*Komega <= M_PI/sqrt(3.0) ) {
						
						// Dispersion relation
						
						EkAA = Lsoc * (4.0*sin(1.5*Kperp)*cos(0.5*sqrt(3.0)*(ik-(Kcut-1.0)/2.0)*Komega) - 2.0*sin(sqrt(3.0)*(ik-(Kcut-1.0)/2.0)*Komega));
						Ek = 4.0*cos(1.5*Kperp)*cos(0.5*sqrt(3.0)*(ik-(Kcut-1.0)/2.0)*Komega) + 2.0*cos(sqrt(3.0)*(ik-(Kcut-1.0)/2.0)*Komega) + 3.0;
			
						// Local Green's function
						
						ReG1 = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0) * Bloch - 0.5*Gap + EkAA;
						ReG2 = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0) * Bloch + 0.5*Gap - EkAA
							- Ek * ReG1 / (pow(ReG1,2) + pow(0.5*Gamma,2));
						ImG2 = 0.5*Gamma * (1.0 + Ek / (pow(ReG1,2) + pow(0.5*Gamma,2)));
						
						FermiDist = 1.0 / (1.0+exp(((iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0) * Bloch)/Temp));
						
						RE1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
							Komega*sqrt(3.0)/(2.0*M_PI) * ReG2 / (pow(ReG2,2) + pow(ImG2,2));
						IM1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
							Komega*sqrt(3.0)/(2.0*M_PI) * (-ImG2) / (pow(ReG2,2) + pow(ImG2,2));
					}
				}
			}
			
			
			// Mod 4 : Zigzag - Finite Field
			
			if (Mod == 4) {
				
				// 1-band retarded lattice Green's function
				
				gsl_matrix_complex *kGreenInv1A = gsl_matrix_complex_alloc (Ncut, Ncut);
				gsl_matrix_complex *kGreenInv1B = gsl_matrix_complex_alloc (Ncut, Ncut);
				
				for (p = 0; p <= Ncut-1; p++)
				{
					for (q = 0; q <= Ncut-1; q++)
					{
						if (p == q) {
							
							FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch + 0.5*Gap;
							
							gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
							
							FloHamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch - 0.5*Gap;
							
							gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (FloHamRe, 0.5*Gamma));
							
						} else if (p == q + 1) {
							
							FloHamRe = -2.0*Lsoc * sin(1.5*Kperp);
							
							gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0));
							gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (-FloHamRe, 0));
							
						} else if (p == q - 1) {
							
							FloHamRe = -2.0*Lsoc * sin(1.5*Kperp);
							
							gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (FloHamRe, 0));
							gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (-FloHamRe, 0));
							
						} else if (p == q + 2) {
							
							gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, -Lsoc));
							gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, Lsoc));
							
						} else if (p == q - 2) {
							
							gsl_matrix_complex_set (kGreenInv1A, p, q, gsl_complex_rect (0, Lsoc));
							gsl_matrix_complex_set (kGreenInv1B, p, q, gsl_complex_rect (0, -Lsoc));
							
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
				gsl_matrix_complex *kGreenInv2B = gsl_matrix_complex_alloc (Ncut, Ncut);
				gsl_complex kGreenP1A, kGreenP2A, kGreenP3A;
				gsl_complex kGreenP4A, kGreenP5A, kGreenP6A;
				gsl_complex kGreenP7A, kGreenP8A, kGreenP9A;
				gsl_complex kGreenP1B, kGreenP2B, kGreenP3B;
				gsl_complex kGreenP4B, kGreenP5B, kGreenP6B;
				gsl_complex kGreenP7B, kGreenP8B, kGreenP9B;
				
				for (p = 0; p <= Ncut-1; p++)
				{
					for (q = 0; q <= Ncut-1; q++)
					{
						if (p == 0 || q == 0) {
							
							kGreenP1A = gsl_complex_rect (0, 0);
							kGreenP1B = gsl_complex_rect (0, 0);
							
						} else {
							
							kGreenP1A = gsl_matrix_complex_get (kGreen1A, p-1, q-1);
							kGreenP1B = gsl_matrix_complex_get (kGreen1B, p-1, q-1);
						}
						
						kGreenP2A = gsl_matrix_complex_get (kGreen1A, p, q);
						kGreenP2B = gsl_matrix_complex_get (kGreen1B, p, q);
						
						if (p == Ncut-1 || q == Ncut-1) {
							
							kGreenP3A = gsl_complex_rect (0, 0);
							kGreenP3B = gsl_complex_rect (0, 0);
							
						} else {
							
							kGreenP3A = gsl_matrix_complex_get (kGreen1A, p+1, q+1);
							kGreenP3B = gsl_matrix_complex_get (kGreen1B, p+1, q+1);
						}
						
						if (p == 0 || q == Ncut-1) {
							
							kGreenP4A = gsl_complex_rect (0, 0);
							kGreenP4B = gsl_complex_rect (0, 0);
							
						} else {
							
							kGreenP4A = gsl_matrix_complex_get (kGreen1A, p-1, q+1);
							kGreenP4B = gsl_matrix_complex_get (kGreen1B, p-1, q+1);
						}
						
						if (p == Ncut-1 || q == 0) {
							
							kGreenP5A = gsl_complex_rect (0, 0);
							kGreenP5B = gsl_complex_rect (0, 0);
							
						} else {
							
							kGreenP5A = gsl_matrix_complex_get (kGreen1A, p+1, q-1);
							kGreenP5B = gsl_matrix_complex_get (kGreen1B, p+1, q-1);
						}
						
						if (q == 0) {
							
							kGreenP6A = gsl_complex_rect (0, 0);
							kGreenP6B = gsl_complex_rect (0, 0);
							
						} else {
							
							kGreenP6A = gsl_matrix_complex_get (kGreen1A, p, q-1);
							kGreenP6B = gsl_matrix_complex_get (kGreen1B, p, q-1);
						}
						
						if (q == Ncut-1) {
							
							kGreenP7A = gsl_complex_rect (0, 0);
							kGreenP7B = gsl_complex_rect (0, 0);
							
						} else {
							
							kGreenP7A = gsl_matrix_complex_get (kGreen1A, p, q+1);
							kGreenP7B = gsl_matrix_complex_get (kGreen1B, p, q+1);
						}
						
						if (p == 0) {
							
							kGreenP8A = gsl_complex_rect (0, 0);
							kGreenP8B = gsl_complex_rect (0, 0);
							
						} else {
							
							kGreenP8A = gsl_matrix_complex_get (kGreen1A, p-1, q);
							kGreenP8B = gsl_matrix_complex_get (kGreen1B, p-1, q);
						}
						
						if (p == Ncut-1) {
							
							kGreenP9A = gsl_complex_rect (0, 0);
							kGreenP9B = gsl_complex_rect (0, 0);
							
						} else {
							
							kGreenP9A = gsl_matrix_complex_get (kGreen1A, p+1, q);
							kGreenP9B = gsl_matrix_complex_get (kGreen1B, p+1, q);
						}
						
						OrbMixARe = GSL_REAL(kGreenP1B) + GSL_REAL(kGreenP2B) + GSL_REAL(kGreenP3B) 
							+ GSL_REAL(kGreenP4B) + GSL_REAL(kGreenP5B)
							+ cos(1.5*Kperp) * (GSL_REAL(kGreenP6B) + GSL_REAL(kGreenP7B))
							- sin(1.5*Kperp) * (GSL_IMAG(kGreenP6B) + GSL_IMAG(kGreenP7B))
							+ cos(1.5*Kperp) * (GSL_REAL(kGreenP8B) + GSL_REAL(kGreenP9B))
							+ sin(1.5*Kperp) * (GSL_IMAG(kGreenP8B) + GSL_IMAG(kGreenP9B));
						
						OrbMixAIm = GSL_IMAG(kGreenP1B) + GSL_IMAG(kGreenP2B) + GSL_IMAG(kGreenP3B) 
							+ GSL_IMAG(kGreenP4B) + GSL_IMAG(kGreenP5B)
							+ cos(1.5*Kperp) * (GSL_IMAG(kGreenP6B) + GSL_IMAG(kGreenP7B))
							+ sin(1.5*Kperp) * (GSL_REAL(kGreenP6B) + GSL_REAL(kGreenP7B))
							+ cos(1.5*Kperp) * (GSL_IMAG(kGreenP8B) + GSL_IMAG(kGreenP9B))
							- sin(1.5*Kperp) * (GSL_REAL(kGreenP8B) + GSL_REAL(kGreenP9B));
						
						OrbMixBRe = GSL_REAL(kGreenP1A) + GSL_REAL(kGreenP2A) + GSL_REAL(kGreenP3A) 
							+ GSL_REAL(kGreenP4A) + GSL_REAL(kGreenP5A)
							+ cos(1.5*Kperp) * (GSL_REAL(kGreenP6A) + GSL_REAL(kGreenP7A))
							+ sin(1.5*Kperp) * (GSL_IMAG(kGreenP6A) + GSL_IMAG(kGreenP7A))
							+ cos(1.5*Kperp) * (GSL_REAL(kGreenP8A) + GSL_REAL(kGreenP9A))
							- sin(1.5*Kperp) * (GSL_IMAG(kGreenP8A) + GSL_IMAG(kGreenP9A));
						
						OrbMixBIm = GSL_IMAG(kGreenP1A) + GSL_IMAG(kGreenP2A) + GSL_IMAG(kGreenP3A) 
							+ GSL_IMAG(kGreenP4A) + GSL_IMAG(kGreenP5A)
							+ cos(1.5*Kperp) * (GSL_IMAG(kGreenP6A) + GSL_IMAG(kGreenP7A))
							- sin(1.5*Kperp) * (GSL_REAL(kGreenP6A) + GSL_REAL(kGreenP7A))
							+ cos(1.5*Kperp) * (GSL_IMAG(kGreenP8A) + GSL_IMAG(kGreenP9A))
							+ sin(1.5*Kperp) * (GSL_REAL(kGreenP8A) + GSL_REAL(kGreenP9A));
						
						if (p == q) {
							
							GInvARe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch - (-0.5*Gap + OrbMixARe);
							GInvAIm = 0.5*Gamma - OrbMixAIm;
							
							GInvBRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch - ( 0.5*Gap + OrbMixBRe);
							GInvBIm = 0.5*Gamma - OrbMixBIm;
							
						} else if (p == q + 1) {
							
							GInvARe = - 2.0*Lsoc * sin(1.5*Kperp) - OrbMixARe;
							GInvAIm = - OrbMixAIm;
							
							GInvBRe = 2.0*Lsoc * sin(1.5*Kperp) - OrbMixBRe;
							GInvBIm = - OrbMixBIm;
							
						} else if (p == q - 1) {
							
							GInvARe = - 2.0*Lsoc * sin(1.5*Kperp) - OrbMixARe;
							GInvAIm = - OrbMixAIm;
							
							GInvBRe = 2.0*Lsoc * sin(1.5*Kperp) - OrbMixBRe;
							GInvBIm = - OrbMixBIm;
							
						} else if (p == q + 2) {
							
							GInvARe = - OrbMixARe;
							GInvAIm = - Lsoc - OrbMixAIm;
							
							GInvBRe = - OrbMixBRe;
							GInvBIm = Lsoc - OrbMixBIm;
							
						} else if (p == q - 2) {
							
							GInvARe = - OrbMixARe;
							GInvAIm = Lsoc - OrbMixAIm;
							
							GInvBRe = - OrbMixBRe;
							GInvBIm = - Lsoc - OrbMixBIm;
							
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
				
				RE1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) = GSL_REAL(kGreenA);
				IM1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) = GSL_IMAG(kGreenA);
				
				// Free the previous allocation
				
				gsl_permutation_free (perm2A);
				gsl_matrix_complex_free (kGreenInv2A);
				gsl_matrix_complex_free (kGreen2A);
				
				gsl_permutation_free (perm2B);
				gsl_matrix_complex_free (kGreenInv2B);
				gsl_matrix_complex_free (kGreen2B);
			}
		}
	}
	
	return LocGreenSub;
}
