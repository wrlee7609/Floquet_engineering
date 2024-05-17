/************************************************************************************************/
/************* [System] Electric field-driven band insulator (hypercubic lattice) ***************/
/************* [Method] Floquet-Keldysh theory / Self-consistent Born approx.     ***************/
/************* [Object] Local dos/ocupation/distribution, DC current density      ***************/
/************* [Programmer] Dr. Woo-Ram Lee (wrlee@kias.re.kr)                    ***************/
/************************************************************************************************/

// Standard libraries

#include <stdio.h>
#include <math.h>

// MPI routines and definitions for parallel computing

#include <mpi.h>

// GSL libraries

#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_ellint.h>

// Define complex-valued vector

#define RE1(z,i)((z)[2*(i)])
#define IM1(z,i)((z)[2*(i)+1])
#define RE2(z,i,j)((z)[(i)][2*(j)])
#define IM2(z,i,j)((z)[(i)][2*(j)+1])
#define RERet(z,i)((z)[3*(i)])
#define IMRet(z,i)((z)[3*(i)+1])
#define IMLes(z,i)((z)[3*(i)+2])

// Mod

int const		Mod   = 2;						// (1) Zero field, (2) Finite field

// Numerical parameters

int const		Ncpu   = 25;					// total number of CPUs

int const		Iband  = Ncpu* 45;				// grid number of frequency domain in a bandwidth
int const		Ncut   =  89;					// grid number of Floquet mode
int const		Fmax   = Ncpu* 13;				// Field/Domega
int const		Imax   = Ncut* 2*(Ncpu-1)* 13;	// grid number of time domain (> Icut) (even)
int const		Icut   = Ncut* Fmax;			// grid number of frequency domain in a full range
double const	Domega = 1.0/Iband;				// grid in frequency domain
double const	Dtime  = 2.0*M_PI/Imax/Domega;	// grid in time domain

int const		Zband  = Ncpu * 1;				// grid number of zeta variable of momentum in a bandwidth, 3 (Zero field), 1 (Finite field)
int const		Zcut   = Zband* 3;				// grid number of zeta variable of momentum [<< (Ncut-1)/2], 3
double const	Zomega = 1.0/Zband;				// grid in frequency domain of Z

int const		Itrmax = 1000000;				// max number of iteration
double const	Tol0   = 0.000001;				// criteria for convergence, 0.001

// Physical parameters

double const	Temp   = 0.000001*Domega;		// Temperature (Zero Temp = 0.000001*Domega)
double const	Field  = Fmax*Domega;			// Electric field
double const	Gab	   = 1.0;					// Hopping strength b/w orbital A & B
double const	Gamma  = 0.01;					// System-bath coupling strength
double const	Vimp   = 0.20;					// Coupling strength to impurity

// Define functions for subroutine

double *SubLocGreen (int rank, int orb, double Dab, double *RetSelf);

// GSL Function: gsl_sf_ellint_E(double phi, double k, gsl_mode_t mode = 0)


/************************************************************************************************/
/************************************** [Main routine] ******************************************/
/************************************************************************************************/

main(int argc, char **argv)
{	
	// Open the saving file
	
	FILE *f1;
	f1 = fopen("BandE029V020D100S","wt");
	
	// Initiate the MPI system
	
	int rank;		// Rank of my CPU
	int source;		// CPU sending message
	int dest = 0;	// CPU receiving message
	int root = 0;	// CPU broadcasting message to all the others
	int tag = 0;	// Indicator distinguishing messages
	
	MPI_Status status;
	MPI_Init (&argc, &argv);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);
	
	// Define local variables
	
	int i, id, iw, ie, n, orb;
	double Dab, Alpha, Tol, Crit, Crit0;
	double LocDos[Icut], OldDos[Icut];
	
	double RetSelfA[2*Icut], OldRetSelfA[2*Icut], NewRetSelfA[2*Icut];
	double RetSelfB[2*Icut], OldRetSelfB[2*Icut], NewRetSelfB[2*Icut];
	
	double *LocGreenSub;
	double RetLocGreenA[2*Icut];
	double RetLocGreenB[2*Icut];
	
	
	/************************************************************************************************/
	/*********************************** [Step 1] Initial guess *************************************/
	/************************************************************************************************/
	
	Crit0 = 0.0;
	
	for (iw = 0; iw <= Icut-1; iw++)
	{
		RE1(RetSelfA,iw) = IM1(RetSelfA,iw) = 0;
		RE1(RetSelfB,iw) = IM1(RetSelfB,iw) = 0;
		OldDos[iw] = 0.1;
		
		Crit0 += pow(OldDos[iw], 2);
	}
	
	for (id = 2; id <= 2; id++) {	// Loop for changing energy gap b/w orbital A & B
		
		Dab = 6.0 + 2.0*id;			// Increase Dab by 2.0 t*
		
		/************************************************************************************************/
		/******************************* [Step 2] SCBA iteration routine ********************************/
		/************************************************************************************************/
		
		if (rank == 0) {
			
			if (Mod == 1) {	// Zero field
				
				printf (">> Temp / t* = %f, Gamma / t* = %f, Vimp / t* = %f, Dab / t* = %f, |e|Ea / t* = %f \n\n", 
						Temp, Gamma, Vimp, Dab, 0.0);
				printf ("Step / Tol_LocDos \n");
			}
			
			if (Mod == 2) {	// Finite field
				
				printf (">> Temp / t* = %f, Gamma / t* = %f, Vimp / t* = %f, Dab / t* = %f, |e|Ea / t* = %f \n\n", 
						Temp, Gamma, Vimp, Dab, Field);
				printf ("Step / Tol_LocDos \n");
			}
		}
		
		for (i = 1; i <= Itrmax; i++) {
			
			// Local Green's function for orbital A
			
			if (rank >= 1 && rank <= Ncpu-1) {
				
				LocGreenSub = SubLocGreen (rank, 0, Dab, RetSelfA);
				MPI_Send (LocGreenSub, 2*Icut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
				
			} else if (rank == 0) {
				
				for (source = 0; source <= Ncpu-1; source++)
				{
					if (source == 0) {
						
						LocGreenSub = SubLocGreen (source, 0, Dab, RetSelfA);
						
					} else {
						
						MPI_Recv (LocGreenSub, 2*Icut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
					}
					
					for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++)
					{
						for (n = 0; n <= Ncut-1; n++)
						{
							// Retarded local Green's function
							
							RE1(RetLocGreenA,iw+n*Fmax) = RE1(LocGreenSub,iw+Fmax*(n-source)/Ncpu);
							IM1(RetLocGreenA,iw+n*Fmax) = IM1(LocGreenSub,iw+Fmax*(n-source)/Ncpu);
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
				}
			}
			
			// Local Green's function for orbital B
			
			if (rank >= 1 && rank <= Ncpu-1) {
				
				LocGreenSub = SubLocGreen (rank, 1, Dab, RetSelfB);
				MPI_Send (LocGreenSub, 2*Icut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
				
			} else if (rank == 0) {
				
				for (source = 0; source <= Ncpu-1; source++)
				{
					if (source == 0) {
						
						LocGreenSub = SubLocGreen (source, 1, Dab, RetSelfB);
						
					} else {
						
						MPI_Recv (LocGreenSub, 2*Icut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
					}
					
					for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++)
					{
						for (n = 0; n <= Ncut-1; n++)
						{
							// Retarded local Green's function
							
							RE1(RetLocGreenB,iw+n*Fmax) = RE1(LocGreenSub,iw+Fmax*(n-source)/Ncpu);
							IM1(RetLocGreenB,iw+n*Fmax) = IM1(LocGreenSub,iw+Fmax*(n-source)/Ncpu);
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
					
					// Local Dos (including only diagonal parts)
					
					LocDos[iw] = - (IM1(RetLocGreenA,iw) + IM1(RetLocGreenB,iw)) / M_PI;
				}
			}
			
			// Broadcast data to the other CPUs
			
			MPI_Bcast (RetSelfA, 2*Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
			MPI_Bcast (RetSelfB, 2*Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
			MPI_Bcast (LocDos, Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
			
			// Convergence test
			
			Crit = 0.0;
			
			for (iw = 0; iw <= Icut-1; iw++)
			{
				Crit += pow(LocDos[iw] - OldDos[iw], 2);
			}
			
			Tol = sqrt(Crit/Crit0);
			
			Crit0 = 0.0;
			
			for (iw = 0; iw <= Icut-1; iw++)
			{
				OldDos[iw] = LocDos[iw];
				Crit0 += pow(OldDos[iw], 2);
			}
			
			if (rank == 0) {
				
				printf ("%d %e \n", i, Tol);
			}
			
			if (i > 1 && Tol < Tol0) {
				
				if (rank == 0) {
					
					printf (">> Converged within %e \n\n", Tol0);
				}
				
				break;
			}
		}
		
		
		/************************************************************************************************/
		/************************************** [Step 3] Save data **************************************/
		/************************************************************************************************/
		
		if (rank == 0) {
			
			if (Mod == 1) {	// Zero field
				
				for (iw = (Icut-1)/2 - 10*Iband; iw <= (Icut-1)/2 + 10*Iband; iw++)
				{				
					fprintf(f1, "%f %f %f %f %f %f %e \n", 
							Temp, Gamma, Vimp, Dab, 0.0, (iw-(Icut-1.0)/2.0)*Domega, 
							LocDos[iw] );
				}
				
				fprintf(f1, "\n");
			}
			
			if (Mod == 2) {	// Finite field
				
				for (iw = (Icut-1)/2 - 10*Iband; iw <= (Icut-1)/2 + 10*Iband; iw++)
				{				
					fprintf(f1, "%f %f %f %f %f %f %e \n", 
							Temp, Gamma, Vimp, Dab, Field, (iw-(Icut-1.0)/2.0)*Domega, 
							LocDos[iw] );
				}
				
				fprintf(f1, "\n");
			}
		}
	}
	
	// Close save files
	
	fclose(f1);
	
	// Finish MPI system
	
	MPI_Finalize();
}


/************************************************************************************************/
/***************************** [Subroutine] Local Green's functions *****************************/
/************************************************************************************************/

double *SubLocGreen (int rank, int orb, double Dab, double *RetSelf)
{
	// Definition of local variables
	
	int iw, ie, n, p, q, r, s;
	double OrbitalMixRe, OrbitalMixIm, ReGJ1, ImGJ1, ReGJ2, ImGJ2;
	double kgreena, kgreenb, kgreenRe, kgreenIm;
	double kGreena, kGreenb, kGreenRe, kGreenIm;
	static double LocGreenSub[2*Icut/Ncpu];
	
	for (iw = Fmax*rank/Ncpu; iw <= Fmax*(rank+1)/Ncpu-1; iw++)
	{
		for (n = 0; n <= Ncut-1; n++)
		{
			RE1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) = 0;
			IM1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) = 0;
			
			if (Mod == 1) {		// Zero field
				
				for (ie = 0; ie <= 2*Zcut; ie++)
				{
					// Retarded lattice Green's function
					
					kgreena = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0)*Field - 0.5*(2.0*orb-1.0)*Dab - (ie-Zcut)*Zomega;
					kgreenb = 0.5*Gamma;
					
					kgreenRe = kgreena/(pow(kgreena,2) + pow(kgreenb,2));
					kgreenIm = -kgreenb/(pow(kgreena,2) + pow(kgreenb,2));
					
					kGreena = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0)*Field + 0.5*(2.0*orb-1.0)*Dab
					- (ie-Zcut)*Zomega - RE1(RetSelf,iw+n*Fmax) - pow(Gab*(ie-Zcut)*Zomega,2) * kgreenRe;
					kGreenb = 0.5*Gamma - IM1(RetSelf,iw+n*Fmax) - pow(Gab*(ie-Zcut)*Zomega,2) * kgreenIm;
					
					kGreenRe = kGreena/(pow(kGreena,2) + pow(kGreenb,2));
					kGreenIm = -kGreenb/(pow(kGreena,2) + pow(kGreenb,2));
					
					// Retarded local Green's function
					
					RE1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
					Zomega * exp(-pow((ie-Zcut)*Zomega,2))/sqrt(M_PI) * kGreenRe;
					IM1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
					Zomega * exp(-pow((ie-Zcut)*Zomega,2))/sqrt(M_PI) * kGreenIm;
				}
				
			} 
			
			if (Mod == 2) {		// Finite field
				
				for (ie = 0; ie <= Zcut-1; ie++)
				{	
					/************************************************************************************************/
					/******************************************* 1 orbital ******************************************/
					/************************************************************************************************/
					
					// Set up retarded lattice Green's function
					
					gsl_matrix_complex *kGreen1orbInv = gsl_matrix_complex_alloc (Ncut, Ncut);
					gsl_matrix_complex *kGreen1orb = gsl_matrix_complex_alloc (Ncut, Ncut);
					gsl_permutation *perm1 = gsl_permutation_alloc (Ncut); 
					
					for (p = 0; p <= Ncut-1; p++)
					{
						for (q = 0; q <= Ncut-1; q++)
						{
							if (p == q) {
								
								gsl_matrix_complex_set (kGreen1orbInv, p, q, gsl_complex_rect (
																							   
									(iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Field - 0.5*(2.0*orb-1.0)*Dab, 0.5*Gamma ));
								
							} else if (p == q + 1 || p == q - 1) {
								
								gsl_matrix_complex_set (kGreen1orbInv, p, q, gsl_complex_rect (-0.5*ie*Zomega, 0));
								
							} else {
								
								gsl_matrix_complex_set (kGreen1orbInv, p, q, gsl_complex_rect (0, 0) );
							}
						}
					}
					
					// Matrix inversion
					
					gsl_linalg_complex_LU_decomp (kGreen1orbInv, perm1, &s); 
					gsl_linalg_complex_LU_invert (kGreen1orbInv, perm1, kGreen1orb); 
					
					// Free the previous allocation
					
					gsl_permutation_free (perm1);
					gsl_matrix_complex_free (kGreen1orbInv);
					
					
					/************************************************************************************************/
					/****************************************** 2 orbitals ******************************************/
					/************************************************************************************************/
					
					// Set up retarded lattice Green's function
					
					gsl_matrix_complex *kGreen2orbInv = gsl_matrix_complex_alloc (Ncut, Ncut);
					gsl_matrix_complex *kGreen2orb = gsl_matrix_complex_alloc (Ncut, Ncut);
					gsl_permutation *perm2 = gsl_permutation_alloc (Ncut);
					gsl_complex kGreenA, kGreenB, kGreenC, kGreenD;
					
					for (p = 0; p <= Ncut-1; p++)
					{
						for (q = 0; q <= Ncut-1; q++)
						{
							if (p == 0 || q == Ncut-1) {
								
								kGreenA = gsl_complex_rect (0, 0);
								
							} else {
								
								kGreenA = gsl_matrix_complex_get (kGreen1orb, p-1, q+1);
							}
							
							if (p == Ncut-1 || q == Ncut-1) {
								
								kGreenB = gsl_complex_rect (0, 0);
								
							} else {
								
								kGreenB = gsl_matrix_complex_get (kGreen1orb, p+1, q+1);
							}
							
							if (p == 0 || q == 0) {
								
								kGreenC = gsl_complex_rect (0, 0);
								
							} else {
								
								kGreenC = gsl_matrix_complex_get (kGreen1orb, p-1, q-1);
							}
							
							if (p == Ncut-1 || q == 0) {
								
								kGreenD = gsl_complex_rect (0, 0);
								
							} else {
								
								kGreenD = gsl_matrix_complex_get (kGreen1orb, p+1, q-1);
							}
							
							OrbitalMixRe = GSL_REAL(kGreenA) + GSL_REAL(kGreenB) + GSL_REAL(kGreenC) + GSL_REAL(kGreenD);
							OrbitalMixIm = GSL_IMAG(kGreenA) + GSL_IMAG(kGreenB) + GSL_IMAG(kGreenC) + GSL_IMAG(kGreenD);
							
							if (p == q) {
								
								gsl_matrix_complex_set (kGreen2orbInv, p, q, gsl_complex_rect (
																							   
									(iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Field + 0.5*(2.0*orb-1.0)*Dab 
									- RE1(RetSelf,iw+p*Fmax) - pow(0.5*Gab*ie*Zomega,2) * OrbitalMixRe, 
									0.5*Gamma - IM1(RetSelf,iw+p*Fmax) - pow(0.5*Gab*ie*Zomega,2) * OrbitalMixIm ));
								
							} else if (p == q + 1 || p == q - 1) {
								
								gsl_matrix_complex_set (kGreen2orbInv, p, q, gsl_complex_rect (
																							   
									-0.5*ie*Zomega - pow(0.5*Gab*ie*Zomega,2) * OrbitalMixRe, 
									- pow(0.5*Gab*ie*Zomega,2) * OrbitalMixIm ));
								
							} else {
								
								gsl_matrix_complex_set (kGreen2orbInv, p, q, gsl_complex_rect (
																							   
									- pow(0.5*Gab*ie*Zomega,2) * OrbitalMixRe, 
									- pow(0.5*Gab*ie*Zomega,2) * OrbitalMixIm ));
							}
						}
					}
					
					// Matrix inversion
					
					gsl_linalg_complex_LU_decomp (kGreen2orbInv, perm2, &s); 
					gsl_linalg_complex_LU_invert (kGreen2orbInv, perm2, kGreen2orb); 
					
					// Free the previous allocation
					
					gsl_permutation_free (perm2);
					gsl_matrix_complex_free (kGreen2orbInv);
					
					// Retarded local Green's function
					
					gsl_complex RetkGreen = gsl_matrix_complex_get (kGreen2orb, n, n);
					
					RE1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
						2.0*M_PI*Zomega*(ie*Zomega)*exp(-pow(ie*Zomega,2))/M_PI * GSL_REAL(RetkGreen);
					
					IM1(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
						2.0*M_PI*Zomega*(ie*Zomega)*exp(-pow(ie*Zomega,2))/M_PI * GSL_IMAG(RetkGreen);
					
					// Free the previous allocation
					
					gsl_matrix_complex_free (kGreen1orb);
					gsl_matrix_complex_free (kGreen2orb);
				}
			}
		}
	}
	
	return LocGreenSub;
}

