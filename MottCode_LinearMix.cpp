/************************************************************************************************/
/************* [System] Electric field-driven Mott insulator (hypercubic lattice) ***************/
/************* [Method] Floquet-Keldysh DMFT / Self-consistent Born approx.       ***************/
/************* [Impurity solver] Iterated perturbation theory (IPT)	              ***************/
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
#include <gsl/gsl_fft_complex.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_sf_bessel.h>

// Define complex-valued vector

#define RE1(z,i)((z)[2*(i)])
#define IM1(z,i)((z)[2*(i)+1])
#define RE2(z,i,j)((z)[(i)][2*(j)])
#define IM2(z,i,j)((z)[(i)][2*(j)+1])
#define RERet(z,i)((z)[3*(i)])
#define IMRet(z,i)((z)[3*(i)+1])
#define IMLes(z,i)((z)[3*(i)+2])

// Mod

int const		Mod   = 3;						// (1) Zero field, (2) Site-wise, (3) Ladder-wise

// Numerical parameters

int const		Ncpu   =  75;					// total number of CPUs

int const		Iband  = Ncpu* 15;				// grid number of frequency domain in a bandwidth
int const		Ncut   =  9;					// grid number of Floquet mode
int const		Fmax   = Ncpu* 45;				// Field/Domega
int const		Imax   = Ncut* 2*(Ncpu-1)* 45;	// grid number of time domain (> Icut) (even)
int const		Icut   = Ncut* Fmax;			// grid number of frequency domain in a full range
double const	Domega = 1.0/Iband;				// grid in frequency domain
double const	Dtime  = 2.0*M_PI/Imax/Domega;	// grid in time domain

int const		Zband  = Ncpu* 1;				// grid number of zeta variable of momentum in a bandwidth, 3 (Zero field), 1 (Finite field)
int const		Zcut   = Zband* 3;				// grid number of zeta variable of momentum [<< (Ncut-1)/2], 3
double const	Zomega = 1.0/Zband;				// grid in frequency domain of Z

int const		Itrmax = 1000000;				// max number of iteration
double const	Alpha  = 0.7;					// mixing ratio for solutions, 0.7
double const	Tol0   = 0.000001;				// criteria for convergence, 0.000001

// Physical parameters

double const	Temp   = 0.000001*Domega;		// Temperature (Zero Temp = 0.000001*Domega)
double const	Field  = Fmax*Domega;			// Electric field

// Define functions for subroutine

double *SubRetSelf (int rank, double U, double *LesWeissIm);
double *SubLesSelf (int rank, double U, double *LesWeissIm);
double *SubLocGreen (int rank, double Gamma, double *RetSelfV, double *LesSelfVIm, double *RetSelfU, double *LesSelfUIm);
double *SubCurrent (int rank, double Gamma, double *RetSelfV, double *LesSelfVIm, double *RetSelfU, double *LesSelfUIm);


/************************************************************************************************/
/************************************** [Main routine] ******************************************/
/************************************************************************************************/

main(int argc, char **argv)
{	
	// Open output files
	
	FILE *f1, *f2;
	f1 = fopen("MottE300V000U000S","wt");
	f2 = fopen("MottE300V000U000J","wt");
	
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
	
	int i, ig, iv, iu, itr, iw, ie, n;
	double Gamma, Vimp, U, Tol, Crit, Crit0;
	double kGreena, kGreenb;
	double LocDos[Icut], TunDos[Icut], MedDos[Icut];
	double LocOcc[Icut], TunOcc[Icut], MedOcc[Icut];
	double LocDist[Icut], TunDist[Icut], MedDist[Icut];
	double FermiDist[Icut];

	double *RetSelfTSub;
	double RetSelfT[2*Imax];
	double RetSelfU[2*Icut];
	
	double *LesSelfTSub;
	double LesSelfT[2*Imax];
	double LesSelfUIm[Icut];
	
	double RetSelfV[2*Icut];
	double LesSelfVIm[Icut];
	
	double *LocGreenSub;
	double RetLocGreen[2*Icut];
	double LesLocGreenIm[Icut];
	
	double SGRe, SGIm;
	double RetWeiss[2*Icut];
	double LesWeissIm[Icut];
	double NewLesWeissIm[Icut];
	double OldLesWeissIm[Icut];
	
	double *CurrentSub;
	double Jdc, Jtun1, Jtun2;
	
	
	/************************************************************************************************/
	/************************************ [Step 1] Initial guess ************************************/
	/************************************************************************************************/
	
	Crit0 = 0.0;
	
	for (iw = 0; iw <= Icut-1; iw++)
	{
		FermiDist[iw] = 1.0/(1.0+exp((iw-(Icut-1.0)/2.0)*Domega/Temp));
		LesWeissIm[iw] = 2.0*M_PI * exp(-pow((iw-(Icut-1.0)/2.0)*Domega,2))/sqrt(M_PI) * FermiDist[iw];
		OldLesWeissIm[iw] = LesWeissIm[iw];
		
		RE1(RetSelfV,iw) = IM1(RetSelfV,iw) = LesSelfVIm[iw] = 0;
		
		Crit0 += pow(OldLesWeissIm[iw], 2);
	}
	
	for (ig = 5; ig <= 5; ig++) {		// Loop for changing system-bath scattering rate, Gamma (1 ~ 5)
	
		for (iv = 0; iv <= 0; iv++) {		// Loop for changing impurity coupling strength, Vimp (0 ~ 5)
			
			for (iu = 0; iu <= 0; iu++) {		// Loop for changing on-site interaction strength, U (0 ~ 140)
				
				Gamma = 0.002*ig;	// Increase Gamma by 0.002 t*
				Vimp = 0.1*iv;		// Increase Vimp by 0.1 t*				
				U = 0.0 + 2.0*iu;	// Increase U by 0.1 t*
				
				/************************************************************************************************/
				/**************************** [Step 2] SCBA + DMFT iteration routine ****************************/
				/************************************************************************************************/
				
				if (rank == 0) {
					
					if (Mod == 1) {	// Zero field
						
						printf (">> Temp / t* = %f, Gamma / t* = %f, Vimp / t* = %f, U / t* = %f, |e|Ea / t* = %f \n\n", 
								Temp, Gamma, Vimp, U, 0.0);
						printf ("Step / Tol_LesWeissIm \n");
					}
					
					if (Mod == 2 || Mod == 3) {	// Finite field
						
						printf (">> Temp / t* = %f, Gamma / t* = %f, Vimp / t* = %f, U / t* = %f, |e|Ea / t* = %f \n\n", 
								Temp, Gamma, Vimp, U, Field);
						printf ("Step / Tol_LesWeissIm \n");
					}
				}
				
				for (i = 1; i <= Itrmax; i++)
				{
					// Self-energy for electron correlation (IPT)
					
					if (rank >= 1 && rank <= Ncpu-2) {		// Even grid
						
						RetSelfTSub = SubRetSelf (rank, U, LesWeissIm);
						LesSelfTSub = SubLesSelf (rank, U, LesWeissIm);
						
						MPI_Send (RetSelfTSub, 2*Imax/(Ncpu-1), MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
						MPI_Send (LesSelfTSub, 2*Imax/(Ncpu-1), MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
						
					} else if (rank == 0) {
						
						for (source = 0; source <= Ncpu-2; source++)	// Even grid
						{
							if (source == 0) {
								
								RetSelfTSub = SubRetSelf (source, U, LesWeissIm);
								LesSelfTSub = SubLesSelf (source, U, LesWeissIm);
								
							} else {
								
								MPI_Recv (RetSelfTSub, 2*Imax/(Ncpu-1), MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
								MPI_Recv (LesSelfTSub, 2*Imax/(Ncpu-1), MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
							}
							
							for (itr = Imax*source/(Ncpu-1); itr <= Imax*(source+1)/(Ncpu-1)-1; itr++)
							{
								RE1(RetSelfT,itr) = RE1(RetSelfTSub,itr-Imax*source/(Ncpu-1));
								IM1(RetSelfT,itr) = IM1(RetSelfTSub,itr-Imax*source/(Ncpu-1));
								
								RE1(LesSelfT,itr) = RE1(LesSelfTSub,itr-Imax*source/(Ncpu-1));
								IM1(LesSelfT,itr) = IM1(LesSelfTSub,itr-Imax*source/(Ncpu-1));
							}
						}
						
						// Fast Fourier transform (FFT)
						
						gsl_fft_complex_wavetable *wavetable1;
						gsl_fft_complex_workspace *workspace1;
						
						wavetable1 = gsl_fft_complex_wavetable_alloc (Imax);
						workspace1 = gsl_fft_complex_workspace_alloc (Imax);
						
						gsl_fft_complex_backward (RetSelfT, 1, Imax, wavetable1, workspace1);
						
						gsl_fft_complex_wavetable_free (wavetable1);
						gsl_fft_complex_workspace_free (workspace1);
						
						// Retarded self-energy in frequency domain
						
						for (iw = 0; iw <= (Icut-1)/2-1; iw++)
						{
							RE1(RetSelfU,iw) = RE1(RetSelfT,iw+Imax-(Icut-1)/2);
							IM1(RetSelfU,iw) = IM1(RetSelfT,iw+Imax-(Icut-1)/2);
						}
						
						for (iw = (Icut-1)/2; iw <= Icut-1; iw++)
						{
							RE1(RetSelfU,iw) = RE1(RetSelfT,iw-(Icut-1)/2);
							IM1(RetSelfU,iw) = IM1(RetSelfT,iw-(Icut-1)/2);
						}
						
						// Fast Fourier transform (FFT)
						
						gsl_fft_complex_wavetable *wavetable2;
						gsl_fft_complex_workspace *workspace2;
						
						wavetable2 = gsl_fft_complex_wavetable_alloc (Imax);
						workspace2 = gsl_fft_complex_workspace_alloc (Imax);
						
						gsl_fft_complex_backward (LesSelfT, 1, Imax, wavetable2, workspace2);
						
						gsl_fft_complex_wavetable_free (wavetable2);
						gsl_fft_complex_workspace_free (workspace2);
						
						// Lesser self-energy in frequency domain
						
						for (iw = 0; iw <= (Icut-1)/2-1; iw++)
						{
							LesSelfUIm[iw] = IM1(LesSelfT,iw+Imax-(Icut-1)/2);
						}
						
						for (iw = (Icut-1)/2; iw <= Icut-1; iw++)
						{
							LesSelfUIm[iw] = IM1(LesSelfT,iw-(Icut-1)/2);
						}
					}
					
					// Broadcast data to other CPUs
					
					MPI_Bcast (RetSelfU, 2*Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
					MPI_Bcast (LesSelfUIm, Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
					
					// Green's functions
					
					if (rank >= 1 && rank <= Ncpu-1) {
						
						LocGreenSub = SubLocGreen (rank, Gamma, RetSelfV, LesSelfVIm, RetSelfU, LesSelfUIm);
						MPI_Send (LocGreenSub, 3*Icut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
						
					} else if (rank == 0) {
						
						for (source = 0; source <= Ncpu-1; source++)
						{
							if (source == 0) {
								
								LocGreenSub = SubLocGreen (source, Gamma, RetSelfV, LesSelfVIm, RetSelfU, LesSelfUIm);
								
							} else {
								
								MPI_Recv (LocGreenSub, 3*Icut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
							}
							
							for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++)
							{
								for (n = 0; n <= Ncut-1; n++)
								{
									// Retarded local Green's function
									
									RE1(RetLocGreen,iw+n*Fmax) = RERet(LocGreenSub,iw+Fmax*(n-source)/Ncpu);
									IM1(RetLocGreen,iw+n*Fmax) = IMRet(LocGreenSub,iw+Fmax*(n-source)/Ncpu);
									
									// Lesser local Green's function
									
									if (Mod == 1 || Mod == 2) {	// Zero & low field
										
										LesLocGreenIm[iw+n*Fmax] = -2.0*IM1(RetLocGreen,iw+n*Fmax)*FermiDist[iw+n*Fmax];
									} 
									
									if (Mod == 3) {	// High field
										
										LesLocGreenIm[iw+n*Fmax] = IMLes(LocGreenSub,iw+Fmax*(n-source)/Ncpu);
									}
								}
							}
						}
						
						for (iw = 0; iw <= Icut-1; iw++)
						{
							// Local Dos / Occupation / Distribution
							
							LocDos[iw] = -IM1(RetLocGreen,iw)/M_PI;
							LocOcc[iw] = LesLocGreenIm[iw]/(2.0*M_PI);
							LocDist[iw] = LocOcc[iw] / LocDos[iw];
							
							// Tunneling Dos / Occupation / Distribution
							
							kGreena = (iw-(Icut-1.0)/2.0)*Domega - RE1(RetSelfV,iw) - RE1(RetSelfU,iw);
							kGreenb = 0.5*Gamma - IM1(RetSelfV,iw) - IM1(RetSelfU,iw);
							
							TunDos[iw] = kGreenb/(pow(kGreena,2) + pow(kGreenb,2))/M_PI;
							TunDist[iw] = (Gamma*FermiDist[iw] + LesSelfVIm[iw] + LesSelfUIm[iw]) 
							/ (Gamma - 2.0*(IM1(RetSelfV,iw) + IM1(RetSelfU,iw)));
							TunOcc[iw] = TunDos[iw] * TunDist[iw];
							
							// Self-energy for impurity averaging (SCBA)
							
							RE1(RetSelfV,iw) = pow(Vimp,2) * RE1(RetLocGreen,iw);
							IM1(RetSelfV,iw) = pow(Vimp,2) * IM1(RetLocGreen,iw);
							LesSelfVIm[iw] = pow(Vimp,2) * LesLocGreenIm[iw];
							
							// Retarded Weiss function
							
							SGRe = 1.0 + RE1(RetSelfU,iw) * RE1(RetLocGreen,iw) - IM1(RetSelfU,iw) * IM1(RetLocGreen,iw);
							SGIm = RE1(RetSelfU,iw) * IM1(RetLocGreen,iw) + IM1(RetSelfU,iw) * RE1(RetLocGreen,iw);
							
							RE1(RetWeiss,iw) = (RE1(RetLocGreen,iw)*SGRe + IM1(RetLocGreen,iw)*SGIm) / (pow(SGRe,2)+pow(SGIm,2));
							IM1(RetWeiss,iw) = (-RE1(RetLocGreen,iw)*SGIm + IM1(RetLocGreen,iw)*SGRe) / (pow(SGRe,2)+pow(SGIm,2));
							
							// Lesser Weiss function
							
							if (Mod == 1 || Mod == 2) {	// Zero & low field
								
								LesWeissIm[iw] = -2.0*IM1(RetWeiss,iw) * FermiDist[iw];
							}
							
							if (Mod == 3) {	// High field
								
								LesWeissIm[iw] = ( pow(RE1(RetWeiss,iw),2) + pow(IM1(RetWeiss,iw),2) )
								* ( LesLocGreenIm[iw] / (pow(RE1(RetLocGreen,iw),2) + pow(IM1(RetLocGreen,iw),2)) - LesSelfUIm[iw] );
							}
							
							// Medium Dos / Occupation / Distribution
							
							MedDos[iw] = -IM1(RetWeiss,iw)/M_PI;
							MedOcc[iw] = LesWeissIm[iw]/(2.0*M_PI);
							MedDist[iw] = MedOcc[iw] / MedDos[iw];
							
							// Linear Mixing (to accelerate the convergence rate)
							
							NewLesWeissIm[iw] = LesWeissIm[iw];
							
							if (i > 1) {
								
								LesWeissIm[iw] = Alpha * NewLesWeissIm[iw] + (1.0-Alpha) * OldLesWeissIm[iw];
							}
						}
					}
					
					// Broadcast data to the other CPUs
					
					MPI_Bcast (RetSelfV, 2*Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
					MPI_Bcast (LesSelfVIm, Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
					MPI_Bcast (LesWeissIm, Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
					MPI_Bcast (NewLesWeissIm, Icut, MPI_DOUBLE, root, MPI_COMM_WORLD);
					
					// Convergence test
					
					Crit = 0.0;
					
					for (iw = 0; iw <= Icut-1; iw++)
					{
						Crit += pow(NewLesWeissIm[iw] - OldLesWeissIm[iw], 2);
					}
					
					Tol = sqrt(Crit/Crit0);
					
					Crit0 = 0.0;
					
					for (iw = 0; iw <= Icut-1; iw++)
					{
						OldLesWeissIm[iw] = LesWeissIm[iw];
						Crit0 += pow(OldLesWeissIm[iw], 2);
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
				/************************************* [Step 3] DC current **************************************/
				/************************************************************************************************/
				
				if (Mod == 1) {		// Zero field
					
					if (rank == 0) {
						
						Jdc = Jtun1 = Jtun2 = 0;
						
						printf ("[DC current] (Exact, Tun1, Tun2) = (%e, %e, %e) \n\n", Jdc, Jtun1, Jtun2);
					}
				}
				
				if (Mod == 2) {		// Site-wise thermalization
					
					if (rank == 0) {
						
						// Tunneling formula
						
						Jtun1 = Jtun2 = 0;
						
						for (iw = 0; iw <= Icut-Fmax-1; iw++)
						{
							Jtun1 += Domega * 0.25 * LocDos[iw]*LocDos[iw+Fmax] * (LocDist[iw]-LocDist[iw+Fmax]);
							Jtun2 += Domega * 0.25 * TunDos[iw]*TunDos[iw+Fmax] * (TunDist[iw]-TunDist[iw+Fmax]);
						}
						
						Jdc = Jtun1;
						
						printf ("[DC current] (Exact, Tun1, Tun2) = (%e, %e, %e) \n\n", Jdc, Jtun1, Jtun2);
					}
				}
				
				if (Mod == 3) {		// Ladder-wise thermalization
					
					if (rank >= 1 && rank <= Ncpu-1) {
						
						CurrentSub = SubCurrent (rank, Gamma, RetSelfV, LesSelfVIm, RetSelfU, LesSelfUIm);
						MPI_Send (CurrentSub, Zcut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
						
					} else if (rank == 0) {
						
						// Exact formula
						
						Jdc = 0;
						
						for (source = 0; source <= Ncpu-1; source++)
						{
							if (source == 0) {
								
								CurrentSub = SubCurrent (source, Gamma, RetSelfV, LesSelfVIm, RetSelfU, LesSelfUIm);
								
							} else {
								
								MPI_Recv (CurrentSub, Zcut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
							}
							
							for (ie = Zcut*source/Ncpu; ie <= Zcut*(source+1)/Ncpu-1; ie++)
							{
								Jdc += CurrentSub[ie-Zcut*source/Ncpu];
							}
						}
						
						// Tunneling formula
						
						Jtun1 = Jtun2 = 0;
						
						for (iw = 0; iw <= Icut-Fmax-1; iw++)
						{
							Jtun1 += Domega * 0.25 * LocDos[iw]*LocDos[iw+Fmax] * (LocDist[iw]-LocDist[iw+Fmax]);
							Jtun2 += Domega * 0.25 * TunDos[iw]*TunDos[iw+Fmax] * (TunDist[iw]-TunDist[iw+Fmax]);
						}
						
						printf ("[DC current] (Exact, Tun1, Tun2) = (%e, %e, %e) \n\n", Jdc, Jtun1, Jtun2);
					}
				}
				
				
				/************************************************************************************************/
				/************************************** [Step 4] Save data **************************************/
				/************************************************************************************************/
				
				if (rank == 0) {
					
					if (Mod == 1) {	// Zero field
						
						for (iw = (Icut-1)/2 - 10*Iband; iw <= (Icut-1)/2 + 10*Iband; iw++)
						{		
							fprintf(f1, "%f %f %f %f %f %f %e %e %e %e %e %e %e %e %e %e %e %e %e %e \n", 
									Temp, Gamma, Vimp, U, 0.0, (iw-(Icut-1.0)/2.0)*Domega, 
									LocDos[iw], LocOcc[iw], FermiDist[iw], 
									LocDos[iw]*LocDos[iw+Fmax], FermiDist[iw]-FermiDist[iw+Fmax], 
									RE1(RetSelfU,iw), IM1(RetSelfU,iw), LesSelfUIm[iw],
									LocDos[iw], LocOcc[iw], FermiDist[iw], 
									MedDos[iw], MedOcc[iw], MedDist[iw]);
						}
						
						fprintf(f1, "\n");
						
						fprintf(f2, "%f %f %f %f %f %e %e %e \n", Temp, Gamma, Vimp, U, 0.0, Jdc, Jtun1, Jtun2);
					}
					
					if (Mod == 2 || Mod == 3) {	// Finite field
						
						for (iw = (Icut-1)/2 - 10*Iband; iw <= (Icut-1)/2 + 10*Iband; iw++)
						{			
							fprintf(f1, "%f %f %f %f %f %f %e %e %e %e %e %e %e %e %e %e %e %e %e %e \n", 
									Temp, Gamma, Vimp, U, Field, (iw-(Icut-1.0)/2.0)*Domega, 
									LocDos[iw], LocOcc[iw], LocDist[iw], 
									LocDos[iw]*LocDos[iw+Fmax], LocDist[iw]-LocDist[iw+Fmax], 
									RE1(RetSelfU,iw), IM1(RetSelfU,iw), LesSelfUIm[iw],
									TunDos[iw], TunOcc[iw], TunDist[iw], 
									MedDos[iw], MedOcc[iw], MedDist[iw]);
						}
						
						fprintf(f1, "\n");
						
						fprintf(f2, "%f %f %f %f %f %e %e %e \n", Temp, Gamma, Vimp, U, Field, Jdc, Jtun1, Jtun2);
					}
				}
			}
		}
	}
	
	// Close save files
	
	fclose(f1);
	fclose(f2);
	
	// Finish MPI system
	
	MPI_Finalize();
}


/************************************************************************************************/
/****************************** [Subroutine] Retarded self-energy *******************************/
/************************************************************************************************/

double *SubRetSelf (int rank, double U, double *LesWeissIm)
{
	// Definition of local variables
	
	int itr, itr0, ie;
	double LesWeissTRe, LesWeissTIm;
	static double RetSelfTSub[2*Imax/(Ncpu-1)];
	
	for (itr = rank*Imax/(Ncpu-1); itr <= (rank+1)*Imax/(Ncpu-1)-1; itr++)
	{
		// Lesser Weiss function in time domain
		
		if (itr <= Imax/2-1) {
			
			itr0 = 0;
			
		} else {
			
			itr0 = Imax;
		}
		
		LesWeissTRe = LesWeissTIm = 0;
		
		for (ie = 0; ie <= Icut-1; ie++)
		{
			LesWeissTRe += Domega/(2.0*M_PI)*LesWeissIm[ie]*sin(2.0*M_PI*(ie-(Icut-1.0)/2.0)*(itr-itr0)/Imax);
			LesWeissTIm += Domega/(2.0*M_PI)*LesWeissIm[ie]*cos(2.0*M_PI*(ie-(Icut-1.0)/2.0)*(itr-itr0)/Imax);
		}
		
		// Retarded self-energy in time domain
		
		if (itr == 0) {
			
			RE1(RetSelfTSub,itr-rank*Imax/(Ncpu-1)) = 0;
			IM1(RetSelfTSub,itr-rank*Imax/(Ncpu-1)) = -Dtime*pow(U,2)*(pow(LesWeissTIm,2)-3.0*pow(LesWeissTRe,2))*LesWeissTIm;
			
		} else if (itr >= 1 && itr <= Imax/2-1) {
			
			RE1(RetSelfTSub,itr-rank*Imax/(Ncpu-1)) = 0;
			IM1(RetSelfTSub,itr-rank*Imax/(Ncpu-1)) = -2.0*Dtime*pow(U,2)*(pow(LesWeissTIm,2)-3.0*pow(LesWeissTRe,2))*LesWeissTIm;
			
		} else {
			
			RE1(RetSelfTSub,itr-rank*Imax/(Ncpu-1)) = 0;
			IM1(RetSelfTSub,itr-rank*Imax/(Ncpu-1)) = 0;
		}
	}
	
	return RetSelfTSub;
}


/************************************************************************************************/
/******************************** [Subroutine] Lesser self-energy *******************************/
/************************************************************************************************/

double *SubLesSelf (int rank, double U, double *LesWeissIm)
{
	// Definition of local variables
	
	int itr, itr0, ie;
	double LesWeissTRe, LesWeissTIm;
	static double LesSelfTSub[2*Imax/(Ncpu-1)];
	
	for (itr = rank*Imax/(Ncpu-1); itr <= (rank+1)*Imax/(Ncpu-1)-1; itr++)
	{
		// Lesser Weiss function in time domain
		
		if (itr <= Imax/2-1) {
			
			itr0 = 0;
			
		} else {
			
			itr0 = Imax;
		}
		
		LesWeissTRe = LesWeissTIm = 0;
		
		for (ie = 0; ie <= Icut-1; ie++)
		{
			LesWeissTRe += Domega/(2.0*M_PI)*LesWeissIm[ie]*sin(2.0*M_PI*(ie-(Icut-1.0)/2.0)*(itr-itr0)/Imax);
			LesWeissTIm += Domega/(2.0*M_PI)*LesWeissIm[ie]*cos(2.0*M_PI*(ie-(Icut-1.0)/2.0)*(itr-itr0)/Imax);
		}
		
		// Lesser self-energy in time domain
		
		RE1(LesSelfTSub,itr-rank*Imax/(Ncpu-1)) = -Dtime*pow(U,2)*(pow(LesWeissTRe,2)-3.0*pow(LesWeissTIm,2))*LesWeissTRe;
		IM1(LesSelfTSub,itr-rank*Imax/(Ncpu-1)) = Dtime*pow(U,2)*(pow(LesWeissTIm,2)-3.0*pow(LesWeissTRe,2))*LesWeissTIm;
	}
	
	return LesSelfTSub;
}


/************************************************************************************************/
/**************************** [Subroutine] Local Green's functions ******************************/
/************************************************************************************************/

double *SubLocGreen (int rank, double Gamma, double *RetSelfV, double *LesSelfVIm, double *RetSelfU, double *LesSelfUIm)
{
	// Definition of local variables
	
	int iw, ie, n, m, p;
	double P1a, P1b, P1[2*Ncut];
	double P2a, P2b, P2[2*Ncut];
	double ReGJ, ImGJ, FermiDist;
	double kGreena, kGreenb, kGreen[2*Ncut];
	static double LocGreenSub[3*Icut/Ncpu];
	
	for (iw = Fmax*rank/Ncpu; iw <= Fmax*(rank+1)/Ncpu-1; iw++)
	{
		for (n = 0; n <= Ncut-1; n++)
		{
			RERet(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) = 0;
			IMRet(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) = 0;
			IMLes(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) = 0;
			
			if (Mod == 1) {	// Zero field
				
				for (ie = 0; ie <= 2*Zcut; ie++)
				{
					// Retarded lattice Green's function
					
					kGreena = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0)*Field 
							  - (ie-Zcut)*Zomega - RE1(RetSelfV,iw+n*Fmax) - RE1(RetSelfU,iw+n*Fmax);
					kGreenb = 0.5*Gamma - IM1(RetSelfV,iw+n*Fmax) - IM1(RetSelfU,iw+n*Fmax);
					
					RE1(kGreen,n) = kGreena/(pow(kGreena,2) + pow(kGreenb,2));
					IM1(kGreen,n) = -kGreenb/(pow(kGreena,2) + pow(kGreenb,2));
					
					// Retarded local Green's function
					
					RERet(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
						Zomega * exp(-pow((ie-Zcut)*Zomega,2))/sqrt(M_PI) * RE1(kGreen,n);
					IMRet(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
						Zomega * exp(-pow((ie-Zcut)*Zomega,2))/sqrt(M_PI) * IM1(kGreen,n);
				}
				
			} else {	// Finite field
				
				for (ie = 0; ie <= Zcut-1; ie++)
				{	
					// Set up function P(-)
					
					P1a = (iw-(Fmax-1.0)/2.0)*Domega - (Ncut-1.0)/2.0*Field - RE1(RetSelfV,iw) - RE1(RetSelfU,iw);
					P1b = 0.5*Gamma - IM1(RetSelfV,iw) - IM1(RetSelfU,iw);
					RE1(P1,0) = P1a/(pow(P1a,2) + pow(P1b,2));
					IM1(P1,0) = -P1b/(pow(P1a,2) + pow(P1b,2));
					
					for (p = 1; p <= Ncut-1; p++)
					{
						P1a = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Field - RE1(RetSelfV,iw+p*Fmax) - RE1(RetSelfU,iw+p*Fmax)  
							- pow(0.5*ie*Zomega,2)*RE1(P1,p-1);
						P1b = 0.5*Gamma - IM1(RetSelfV,iw+p*Fmax) - IM1(RetSelfU,iw+p*Fmax) - pow(0.5*ie*Zomega,2)*IM1(P1,p-1);
						RE1(P1,p) = P1a/(pow(P1a,2) + pow(P1b,2));
						IM1(P1,p) = -P1b/(pow(P1a,2) + pow(P1b,2));
					}
					
					// Set up function P(+)
					
					P2a = (iw-(Fmax-1.0)/2.0)*Domega + (Ncut-1.0)/2.0*Field - RE1(RetSelfV,iw+(Ncut-1)*Fmax) - RE1(RetSelfU,iw+(Ncut-1)*Fmax);
					P2b = 0.5*Gamma - IM1(RetSelfV,iw+(Ncut-1)*Fmax) - IM1(RetSelfU,iw+(Ncut-1)*Fmax);
					RE1(P2,Ncut-1) = P2a/(pow(P2a,2) + pow(P2b,2));
					IM1(P2,Ncut-1) = -P2b/(pow(P2a,2) + pow(P2b,2));
					
					for (p = Ncut-2; p >= 0; p--)
					{	
						P2a = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Field - RE1(RetSelfV,iw+p*Fmax) - RE1(RetSelfU,iw+p*Fmax)  
							- pow(0.5*ie*Zomega,2)*RE1(P2,p+1);
						P2b = 0.5*Gamma - IM1(RetSelfV,iw+p*Fmax) - IM1(RetSelfU,iw+p*Fmax) - pow(0.5*ie*Zomega,2)*IM1(P2,p+1);
						RE1(P2,p) = P2a/(pow(P2a,2) + pow(P2b,2));
						IM1(P2,p) = -P2b/(pow(P2a,2) + pow(P2b,2));
					}
					
					// Set up retarded lattice Green's function
					
					if (n == 0) {
						
						kGreena = (iw-(Fmax-1.0)/2.0)*Domega - (Ncut-1.0)/2.0*Field - RE1(RetSelfV,iw) - RE1(RetSelfU,iw) 
								- pow(0.5*ie*Zomega,2)*RE1(P2,1);
						kGreenb = 0.5*Gamma - IM1(RetSelfV,iw) - IM1(RetSelfU,iw) - pow(0.5*ie*Zomega,2)*IM1(P2,1);
						
					} else if (n == Ncut-1) {
						
						kGreena = (iw-(Fmax-1.0)/2.0)*Domega + (Ncut-1.0)/2.0*Field - RE1(RetSelfV,iw+(Ncut-1)*Fmax) - RE1(RetSelfU,iw+(Ncut-1)*Fmax)  
								- pow(0.5*ie*Zomega,2)*RE1(P1,Ncut-2);
						kGreenb = 0.5*Gamma - IM1(RetSelfV,iw+(Ncut-1)*Fmax) - IM1(RetSelfU,iw+(Ncut-1)*Fmax) - pow(0.5*ie*Zomega,2)*IM1(P1,Ncut-2);
						
					} else {
						
						kGreena = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0)*Field - RE1(RetSelfV,iw+n*Fmax) - RE1(RetSelfU,iw+n*Fmax) 
								- pow(0.5*ie*Zomega,2)*(RE1(P1,n-1) + RE1(P2,n+1));
						kGreenb = 0.5*Gamma - IM1(RetSelfV,iw+n*Fmax) - IM1(RetSelfU,iw+n*Fmax) - pow(0.5*ie*Zomega,2)*(IM1(P1,n-1) + IM1(P2,n+1));
					}
					
					// Diagonal elements (m = n)
					
					RE1(kGreen,n) = kGreena/(pow(kGreena,2) + pow(kGreenb,2));
					IM1(kGreen,n) = -kGreenb/(pow(kGreena,2) + pow(kGreenb,2));
					
					// Off-diagonal elements (m < n)
					
					for (m = n-1; m >= 0; m--)
					{
						RE1(kGreen,m) = 0.5*ie*Zomega*(RE1(P1,m)*RE1(kGreen,m+1) - IM1(P1,m)*IM1(kGreen,m+1));
						IM1(kGreen,m) = 0.5*ie*Zomega*(RE1(P1,m)*IM1(kGreen,m+1) + IM1(P1,m)*RE1(kGreen,m+1));
					}
					
					// Off-diagonal elements (m > n)
					
					for (m = n+1; m <= Ncut-1; m++)
					{
						RE1(kGreen,m) = 0.5*ie*Zomega*(RE1(P2,m)*RE1(kGreen,m-1) - IM1(P2,m)*IM1(kGreen,m-1));
						IM1(kGreen,m) = 0.5*ie*Zomega*(RE1(P2,m)*IM1(kGreen,m-1) + IM1(P2,m)*RE1(kGreen,m-1));
					}
					
					// Retarded local Green's function
					
					RERet(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
						2.0*M_PI*Zomega*(ie*Zomega)*exp(-pow(ie*Zomega,2))/M_PI * RE1(kGreen,n);
					IMRet(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
						2.0*M_PI*Zomega*(ie*Zomega)*exp(-pow(ie*Zomega,2))/M_PI * IM1(kGreen,n);
					
					if (Mod == 3) {	// High field
					
						// Lesser local Green's function
						
						for (m = 0; m <= Ncut-1; m++)
						{
							ReGJ = ImGJ = 0;
							
							for (p = 0; p <= Ncut-1; p++)
							{
								ReGJ += RE1(kGreen,p) * gsl_sf_bessel_Jn(p-m,ie*Zomega/Field);
								ImGJ += IM1(kGreen,p) * gsl_sf_bessel_Jn(p-m,ie*Zomega/Field);
							}
							
							FermiDist = 1.0 / (1.0+exp(((iw-(Fmax-1.0)/2.0)*Domega+(m-(Ncut-1.0)/2.0)*Field)/Temp));
							
							IMLes(LocGreenSub,iw+Fmax*(n-rank)/Ncpu) += 
								2.0*M_PI*Zomega*(ie*Zomega)*exp(-pow(ie*Zomega,2))/M_PI
								* ( (pow(ReGJ,2) + pow(ImGJ,2)) * 2.0*0.5*Gamma * FermiDist
								+ (pow(RE1(kGreen,m),2) + pow(IM1(kGreen,m),2)) 
								* (LesSelfVIm[iw+m*Fmax] + LesSelfUIm[iw+m*Fmax]) );
						}
					}
				}
			}
		}
	}
	
	return LocGreenSub;
}


/************************************************************************************************/
/******************************** [Subroutine] Electric current *********************************/
/************************************************************************************************/

double *SubCurrent (int rank, double Gamma, double *RetSelfV, double *LesSelfVIm, double *RetSelfU, double *LesSelfUIm)
{
	// Definition of local variables
	
	int iw, ie, n, m, p;
	double P1a, P1b, P1[2*Ncut];
	double P2a, P2b, P2[2*Ncut];
	double kGreena, kGreenb, kGreen[Ncut][2*Ncut];
	double ReGJ1, ImGJ1, ReGJ2, ImGJ2, FermiDist;
	static double CurrentSub[Zcut/Ncpu];
	
	for (ie = Zcut*rank/Ncpu; ie <= Zcut*(rank+1)/Ncpu-1; ie++)
	{
		CurrentSub[ie-Zcut*rank/Ncpu] = 0;

		for (iw = 0; iw <= Fmax-1; iw++)
		{
			// Set up function P(-)
			
			P1a = (iw-(Fmax-1.0)/2.0)*Domega - (Ncut-1.0)/2.0*Field - RE1(RetSelfV,iw) - RE1(RetSelfU,iw);
			P1b = 0.5*Gamma - IM1(RetSelfV,iw) - IM1(RetSelfU,iw);
			RE1(P1,0) = P1a/(pow(P1a,2) + pow(P1b,2));
			IM1(P1,0) = -P1b/(pow(P1a,2) + pow(P1b,2));
			
			for (n = 1; n <= Ncut-1; n++)
			{
				P1a = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0)*Field - RE1(RetSelfV,iw+n*Fmax) - RE1(RetSelfU,iw+n*Fmax)
					  - pow(0.5*ie*Zomega,2)*RE1(P1,n-1);
				P1b = 0.5*Gamma - IM1(RetSelfV,iw+n*Fmax) - IM1(RetSelfU,iw+n*Fmax) - pow(0.5*ie*Zomega,2)*IM1(P1,n-1);
				RE1(P1,n) = P1a/(pow(P1a,2) + pow(P1b,2));
				IM1(P1,n) = -P1b/(pow(P1a,2) + pow(P1b,2));
			}
			
			// Set up function P(+)
			
			P2a = (iw-(Fmax-1.0)/2.0)*Domega + (Ncut-1.0)/2.0*Field - RE1(RetSelfV,iw+(Ncut-1)*Fmax) - RE1(RetSelfU,iw+(Ncut-1)*Fmax);
			P2b = 0.5*Gamma - IM1(RetSelfV,iw+(Ncut-1)*Fmax) - IM1(RetSelfU,iw+(Ncut-1)*Fmax);
			RE1(P2,Ncut-1) = P2a/(pow(P2a,2) + pow(P2b,2));
			IM1(P2,Ncut-1) = -P2b/(pow(P2a,2) + pow(P2b,2));
			
			for (n = Ncut-2; n >= 0; n--)
			{	
				P2a = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0)*Field - RE1(RetSelfV,iw+n*Fmax) - RE1(RetSelfU,iw+n*Fmax)
					  - pow(0.5*ie*Zomega,2)*RE1(P2,n+1);
				P2b = 0.5*Gamma - IM1(RetSelfV,iw+n*Fmax) - IM1(RetSelfU,iw+n*Fmax) - pow(0.5*ie*Zomega,2)*IM1(P2,n+1);
				RE1(P2,n) = P2a/(pow(P2a,2) + pow(P2b,2));
				IM1(P2,n) = -P2b/(pow(P2a,2) + pow(P2b,2));
			}
			
			// Retarded lattice Green's function
			
			for (n = 0; n <= Ncut-1; n++)
			{
				// Diagonal elements
				
				if (n == 0) {
					
					kGreena = (iw-(Fmax-1.0)/2.0)*Domega - (Ncut-1.0)/2.0*Field - RE1(RetSelfV,iw) - RE1(RetSelfU,iw) 
							  - pow(0.5*ie*Zomega,2)*RE1(P2,1);
					kGreenb = 0.5*Gamma - IM1(RetSelfV,iw) - IM1(RetSelfU,iw) - pow(0.5*ie*Zomega,2)*IM1(P2,1);
					
				} else if (n == Ncut-1) {
					
					kGreena = (iw-(Fmax-1.0)/2.0)*Domega + (Ncut-1.0)/2.0*Field - RE1(RetSelfV,iw+(Ncut-1)*Fmax) - RE1(RetSelfU,iw+(Ncut-1)*Fmax)
							  - pow(0.5*ie*Zomega,2)*RE1(P1,Ncut-2);
					kGreenb = 0.5*Gamma - IM1(RetSelfV,iw+(Ncut-1)*Fmax) - IM1(RetSelfU,iw+(Ncut-1)*Fmax) - pow(0.5*ie*Zomega,2)*IM1(P1,Ncut-2);
					
				} else {
					
					kGreena = (iw-(Fmax-1.0)/2.0)*Domega + (n-(Ncut-1.0)/2.0)*Field - RE1(RetSelfV,iw+n*Fmax) - RE1(RetSelfU,iw+n*Fmax)
							  - pow(0.5*ie*Zomega,2)*(RE1(P1,n-1) + RE1(P2,n+1));
					kGreenb = 0.5*Gamma - IM1(RetSelfV,iw+n*Fmax) - IM1(RetSelfU,iw+n*Fmax) - pow(0.5*ie*Zomega,2)*(IM1(P1,n-1)+IM1(P2,n+1));
				}
				
				RE2(kGreen,n,n) = kGreena/(pow(kGreena,2) + pow(kGreenb,2));
				IM2(kGreen,n,n) = -kGreenb/(pow(kGreena,2) + pow(kGreenb,2));

				// Off-diagonal elements (m < n)
								
				for (m = n-1; m >= 0; m--)
				{
					RE2(kGreen,m,n) = 0.5*ie*Zomega*(RE1(P1,m)*RE2(kGreen,m+1,n)-IM1(P1,m)*IM2(kGreen,m+1,n));
					IM2(kGreen,m,n) = 0.5*ie*Zomega*(RE1(P1,m)*IM2(kGreen,m+1,n)+IM1(P1,m)*RE2(kGreen,m+1,n));
				}

				// Off-diagonal elements (m > n)
				
				for (m = n+1; m <= Ncut-1; m++)
				{
					RE2(kGreen,m,n) = 0.5*ie*Zomega*(RE1(P2,m)*RE2(kGreen,m-1,n)-IM1(P2,m)*IM2(kGreen,m-1,n));
					IM2(kGreen,m,n) = 0.5*ie*Zomega*(RE1(P2,m)*IM2(kGreen,m-1,n)+IM1(P2,m)*RE2(kGreen,m-1,n));
				}
			}
			 
			// Fourier component of current
			
			for (n = 0; n <= Ncut-2; n++)	// for s = 1
			{
				for (m = 0; m <= Ncut-1; m++)
				{
					ReGJ1 = ImGJ1 = ReGJ2 = ImGJ2 = 0;
					
					for (p = 0; p <= Ncut-1; p++)
					{
						ReGJ1 += RE2(kGreen,n+1,p) * gsl_sf_bessel_Jn(p-m,ie*Zomega/Field);
						ImGJ1 += IM2(kGreen,n+1,p) * gsl_sf_bessel_Jn(p-m,ie*Zomega/Field);
						ReGJ2 += RE2(kGreen,n,p) * gsl_sf_bessel_Jn(p-m,ie*Zomega/Field);
						ImGJ2 += IM2(kGreen,n,p) * gsl_sf_bessel_Jn(p-m,ie*Zomega/Field);
					}
					
					FermiDist = 1.0 / (1.0+exp(((iw-(Fmax-1.0)/2.0)*Domega+(m-(Ncut-1.0)/2.0)*Field)/Temp));
					
					CurrentSub[ie-Zcut*rank/Ncpu] += 
						(Domega/2.0/M_PI) * Zomega*pow(ie*Zomega,2) * exp(-pow(ie*Zomega,2))/M_PI
						* ( (ReGJ1 * ImGJ2 - ImGJ1 * ReGJ2) * Gamma * FermiDist
					    + (RE2(kGreen,n+1,m)*IM2(kGreen,n,m) - IM2(kGreen,n+1,m)*RE2(kGreen,n,m))
						* (LesSelfVIm[iw+m*Fmax] + LesSelfUIm[iw+m*Fmax]) );
				}
			}
		}
	}
	
	return CurrentSub;
}

