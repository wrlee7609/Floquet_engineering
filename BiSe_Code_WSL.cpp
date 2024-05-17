/************************************************************************************************/
/************* [System] 3D Topological insulator (Strong, BiSe)                   ***************/
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

// Define 4 orbitals

#define OrbA(z,i)((z)[4*(i)])
#define OrbB(z,i)((z)[4*(i)+1])
#define OrbC(z,i)((z)[4*(i)+2])
#define OrbD(z,i)((z)[4*(i)+3])
 
// Numerical constants

int const       Mod     = 2;                // 1: kx-ky 2D subsystems under E=Ex
                                            // 2: ky-kz 2D subsystems under E=Ey

int	const		Ncpu    = 15;				// total number of CPUs

int const		Iband   = Ncpu * 45;		// grid number of frequency domain in a bandwidth
int const		Ncut    = 75;				// grid number of Floquet mode (75, 261)
int const		Fmax    = Ncpu * 7;			// Field/Domega
int const		Icut    = Ncut * Fmax;		// grid number of frequency domain in a full range
double const	Domega  = 1.0/Iband;		// grid in frequency domain

int const		Kcut    = Ncpu * 7;			// grid number of k_perp

int const		K1band	= Ncpu * 55;		// grid number of k_para within 1 B.W.
int const		K1cut   = K1band * 9;		// grid number of k_para
double const	K1omega = 1.0/K1band;		// grid in k_para domain

// Physical constants (in unit 4D1)

double const	Field   = Fmax*1.0/Iband;	// electric field
double const	Bloch   = Fmax*Domega;		// Bloch frequency

double const	Temp    = 0.000001*Domega;	// temperature (Zero Temp = 0.000001*Domega)
double const	Gamma   = 0.01;             // system-bath scattering rate, 0.01
double const	A1		= 0.6;
double const	A2		= 0.5;
double const	B1		= 0.6;
double const	B2		= 0.3;
double const	D2		= 0.2;
double const    C       = 0;
double const	M		= -0.3;

double const    Kplane  = 1.0 * M_PI;       // perpendicular momentum to the 2D subsystems
                                            // 0.0, 0.1, 0.2, 0.4, 0.6, 1.0

// Define subroutine

double *SubSemiLocDos (int rank, double Kperp);


/************************************************************************************************/
/************************************** [Main routine] ******************************************/
/************************************************************************************************/

main(int argc, char **argv)
{	
	// Open the saving file
	
	FILE *f1;
//	f1 = fopen("BiSe_Ex_16_kz_01_ky","wt");
    f1 = fopen("BiSe_Ey_16_kx_10_kz_Pt4","wt");
	
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
	double Kperp, d1, d2, d3, d4, d5, ek, Ek1;
	double *SemiLocDosSub;
	double SemiLocDos[4*Icut];
	
	// Loop for the lateral momentum

//    for (ik = 0; ik <= (Kcut-1)/4-1; ik++)                // Pt 1
//    for (ik = (Kcut-1)/4; ik <= (Kcut-1)/2-1; ik++)       // Pt 2
//    for (ik = (Kcut-1)/2; ik <= 3*(Kcut-1)/4-1; ik++)     // Pt 3
    for (ik = 3*(Kcut-1)/4; ik <= Kcut-1; ik++)           // Pt 4
    {
		// Set up the lateral momentum
		
		Kperp = (ik-(Kcut-1.0)/2.0)*2.0/Kcut * M_PI;
		
		// Find the semi-local Dos
		
		if (rank >= 1 && rank <= Ncpu-1) {
			
			SemiLocDosSub = SubSemiLocDos (rank, Kperp);
			MPI_Send (SemiLocDosSub, 4*Icut/Ncpu, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD);
			
		} else if (rank == 0) {
			
			for (source = 0; source <= Ncpu-1; source++)
			{
				if (source == 0) {
					
					SemiLocDosSub = SubSemiLocDos (source, Kperp);
					
				} else {
					
					MPI_Recv (SemiLocDosSub, 4*Icut/Ncpu, MPI_DOUBLE, source, tag, MPI_COMM_WORLD, &status);
				}
				
				for (iw = Fmax*source/Ncpu; iw <= Fmax*(source+1)/Ncpu-1; iw++) 
				{
					for (n = 0; n <= Ncut-1; n++)
					{
						OrbA(SemiLocDos,iw+n*Fmax) = OrbA(SemiLocDosSub,iw+Fmax*(n-source)/Ncpu);
                        OrbB(SemiLocDos,iw+n*Fmax) = OrbB(SemiLocDosSub,iw+Fmax*(n-source)/Ncpu);
                        OrbC(SemiLocDos,iw+n*Fmax) = OrbC(SemiLocDosSub,iw+Fmax*(n-source)/Ncpu);
                        OrbD(SemiLocDos,iw+n*Fmax) = OrbD(SemiLocDosSub,iw+Fmax*(n-source)/Ncpu);
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
					
                    if (Mod == 1) {
                        
                        d1 = A1 * sin((ik1-(K1cut-1.0)/2.0)*K1omega);
                        d2 = A1 * sin(Kperp);
                        d3 = M + 2.0 * B1 * (2.0 - cos((ik1-(K1cut-1.0)/2.0)*K1omega) - cos(Kperp)) + 2.0 * B2 * (1.0 - cos(Kplane));
                        d4 = A2 * sin(Kplane);
                        d5 = 0;
                        ek = C + 0.5 * (2.0 - cos((ik1-(K1cut-1.0)/2.0)*K1omega) - cos(Kperp)) + 2.0 * D2 * (1.0 - cos(Kplane));
                    }
                    
                    if (Mod == 2) {
                        
                        d1 = A1 * sin(Kplane);
                        d2 = A1 * sin((ik1-(K1cut-1.0)/2.0)*K1omega);
                        d3 = M + 2.0 * B1 * (2.0 - cos(Kplane) - cos((ik1-(K1cut-1.0)/2.0)*K1omega)) + 2.0 * B2 * (1.0 - cos(Kperp));
                        d4 = A2 * sin(Kperp);
                        d5 = 0;
                        ek = C + 0.5 * (2.0 - cos(Kplane) - cos((ik1-(K1cut-1.0)/2.0)*K1omega)) + 2.0 * D2 * (1.0 - cos(Kperp));
                    }
                    
                    Ek1 += K1omega/(2.0*M_PI) * (ek - sqrt(pow(d1,2) + pow(d2,2) + pow(d3,2) + pow(d4,2) + pow(d5,2)));
				}
			}
		}
		
		// Save data on output files
		
		if (rank == 0) {
			
			for (iw = 0; iw <= Icut-1; iw++)
			{
				fprintf(f1, "%f %f %e %e %e %e\n",
						(ik-(Kcut-1.0)/2.0)*2.0/Kcut, 
						(iw-(Icut-1.0)/2.0)*Domega - Ek1, 
						OrbA(SemiLocDos,iw),
                        OrbB(SemiLocDos,iw),
                        OrbC(SemiLocDos,iw),
                        OrbD(SemiLocDos,iw));
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
/********************************* [Subroutine] Semi-Local Dos **********************************/
/************************************************************************************************/

double *SubSemiLocDos (int rank, double Kperp)
{
	// Definition of local variables
	
	int iw, n, p, q, r, s, perm;
    
    gsl_complex Ppab, Ppamb, Ppapb, Ppabm, Ppabp, Ppambm, Ppapbp, Ppambp, Ppapbm;
    gsl_complex PpLab, PpLabm, PpLabp;
    gsl_complex PpRab, PpRamb, PpRapb;
    
    gsl_complex Pmab, Pmamb, Pmapb, Pmabm, Pmabp, Pmambm, Pmapbp, Pmambp, Pmapbm;
    gsl_complex PmLab, PmLamb, PmLapb;
    gsl_complex PmRab, PmRabm, PmRabp;
    
    gsl_complex Qaab, Qaamb, Qaapb, Qaabm, Qaabp, Qaambm, Qaapbp, Qaambp, Qaapbm;
    gsl_complex QaLab, QaRab;
    
    gsl_complex Qbpab, Qbpamb, Qbpapb, Qbpabm, Qbpabp, Qbpambm, Qbpapbp, Qbpambp, Qbpapbm;
    gsl_complex QbpLab, QbpRab;
    
    gsl_complex Qbmab, Qbmamb, Qbmapb, Qbmabm, Qbmabp, Qbmambm, Qbmapbp, Qbmambp, Qbmapbm;
    gsl_complex QbmLab, QbmRab;
    
    gsl_complex Rapab, Ramab, Rbpab, Rbmab;
    gsl_complex MaLab, MaRab, MbLab, MbRab, McLab, McRab, MdLab, MdRab;
    
    double MaLRe, MaLIm, MaRRe, MaRIm, MbLRe, MbLIm, MbRRe, MbRIm, McLRe, McLIm, McRRe, McRIm, MdLRe, MdLIm, MdRRe, MdRIm;
    
    double MixQaRe, MixQaIm, MixQbpRe, MixQbpIm, MixQbmRe, MixQbmIm;
	double MixRapRe, MixRapIm, MixRamRe, MixRamIm, MixRbpRe, MixRbpIm, MixRbmRe, MixRbmIm;
    double MixGaRe, MixGaIm, MixGbRe, MixGbIm, MixGcRe, MixGcIm, MixGdRe, MixGdIm;
    
    double InvPpRe, InvPpIm, InvPmRe, InvPmIm;
    double InvQaRe, InvQaIm, InvQbpRe, InvQbpIm, InvQbmRe, InvQbmIm;
    double InvRapRe, InvRapIm, InvRamRe, InvRamIm, InvRbpRe, InvRbpIm, InvRbmRe, InvRbmIm;
    double InvGaRe, InvGaIm, InvGbRe, InvGbIm, InvGcRe, InvGcIm, InvGdRe, InvGdIm;
	
    static double SemiLocDosSub[4*Icut/Ncpu];
    
	// Find the semi-local Dos at low field
	
    for (iw = Fmax*rank/Ncpu; iw <= Fmax*(rank+1)/Ncpu-1; iw++)
    {
        for (n = 0; n <= Ncut-1; n++)
        {
            /**************************************/
            /******* xy subsystems under Ex *******/
            /**************************************/
            
            if (Mod == 1) {
                
                /************************************************/
                /*** 1-band retarded lattice Green's function ***/
                /************************************************/
                
                gsl_matrix_complex *InvPp = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvPm = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        if (p == q) {
                            
                            InvPpRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                      - (C + M) - 2.0*(D2 + B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 + B1) * (2.0 - cos(Kperp));
                            InvPpIm = 0.5*Gamma;
                            
                            InvPmRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                      - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 - B1) * (2.0 - cos(Kperp));
                            InvPmIm = 0.5*Gamma;
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            InvPpRe = 0.25 + B1;
                            InvPpIm = 0;
                            
                            InvPmRe = 0.25 - B1;
                            InvPmIm = 0;
                            
                        } else {
                            
                            InvPpRe = InvPpIm = 0;
                            InvPmRe = InvPmIm = 0;
                        }
                        
                        gsl_matrix_complex_set (InvPp, p, q, gsl_complex_rect (InvPpRe, InvPpIm));
                        gsl_matrix_complex_set (InvPm, p, q, gsl_complex_rect (InvPmRe, InvPmIm));
                    }
                }
                
                // Matrix inversion (1 band)
                
                gsl_permutation *permPp = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Pp = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvPp, permPp, &perm);
                gsl_linalg_complex_LU_invert (InvPp, permPp, Pp);
                
                gsl_permutation *permPm = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Pm = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvPm, permPm, &perm);
                gsl_linalg_complex_LU_invert (InvPm, permPm, Pm);
                
                // Free the previous allocation
                
                gsl_permutation_free (permPp);
                gsl_matrix_complex_free (InvPp);
                
                gsl_permutation_free (permPm);
                gsl_matrix_complex_free (InvPm);
                
                
                /************************************************/
                /*** 2-band retarded lattice Green's function ***/
                /************************************************/
                
                gsl_matrix_complex *InvQa = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvQbp = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvQbm = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        Pmab = gsl_matrix_complex_get (Pm, p, q);
                        Ppab = gsl_matrix_complex_get (Pp, p, q);
                        
                        if (p == 0) {
                            
                            Ppamb = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppamb = gsl_matrix_complex_get (Pp, p-1, q);
                        }
                        
                        if (p == Ncut-1) {
                            
                            Ppapb = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppapb = gsl_matrix_complex_get (Pp, p+1, q);
                        }
                        
                        if (q == 0) {
                            
                            Ppabm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppabm = gsl_matrix_complex_get (Pp, p, q-1);
                        }
                        
                        if (q == Ncut-1) {
                            
                            Ppabp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppabp = gsl_matrix_complex_get (Pp, p, q+1);
                        }
                        
                        if (p == 0 || q == 0) {
                            
                            Ppambm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppambm = gsl_matrix_complex_get (Pp, p-1, q-1);
                        }
                        
                        if (p == Ncut-1 || q == Ncut-1) {
                            
                            Ppapbp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppapbp = gsl_matrix_complex_get (Pp, p+1, q+1);
                        }
                        
                        if (p == 0 || q == Ncut-1) {
                            
                            Ppambp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppambp = gsl_matrix_complex_get (Pp, p-1, q+1);
                        }
                        
                        if (p == Ncut-1 || q == 0) {
                            
                            Ppapbm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppapbm = gsl_matrix_complex_get (Pp, p+1, q-1);
                        }
                        
                        MixQaRe = pow(A2*sin(Kplane),2) * GSL_REAL(Pmab);
                        MixQaIm = pow(A2*sin(Kplane),2) * GSL_IMAG(Pmab);
                        
                        MixQbpRe = pow(0.5*A1,2) * ( pow(2.0*sin(Kperp),2) * GSL_REAL(Ppab)
                                   + 2.0*sin(Kperp) * ( GSL_REAL(Ppamb) - GSL_REAL(Ppapb) + GSL_REAL(Ppabm) - GSL_REAL(Ppabp) )
                                   + GSL_REAL(Ppambm) + GSL_REAL(Ppapbp) - GSL_REAL(Ppambp) - GSL_REAL(Ppapbm) );
                        MixQbpIm = pow(0.5*A1,2) * ( pow(2.0*sin(Kperp),2) * GSL_IMAG(Ppab)
                                   + 2.0*sin(Kperp) * ( GSL_IMAG(Ppamb) - GSL_IMAG(Ppapb) + GSL_IMAG(Ppabm) - GSL_IMAG(Ppabp) )
                                   + GSL_IMAG(Ppambm) + GSL_IMAG(Ppapbp) - GSL_IMAG(Ppambp) - GSL_IMAG(Ppapbm) );
                        
                        MixQbmRe = pow(0.5*A1,2) * ( pow(2.0*sin(Kperp),2) * GSL_REAL(Ppab)
                                   + 2.0*sin(Kperp) * ( GSL_REAL(Ppapb) - GSL_REAL(Ppamb) + GSL_REAL(Ppabp) - GSL_REAL(Ppabm) )
                                   + GSL_REAL(Ppapbp) + GSL_REAL(Ppambm) - GSL_REAL(Ppapbm) - GSL_REAL(Ppambp) );
                        MixQbmIm = pow(0.5*A1,2) * ( pow(2.0*sin(Kperp),2) * GSL_IMAG(Ppab)
                                   + 2.0*sin(Kperp) * ( GSL_IMAG(Ppapb) - GSL_IMAG(Ppamb) + GSL_IMAG(Ppabp) - GSL_IMAG(Ppabm) )
                                   + GSL_IMAG(Ppapbp) + GSL_IMAG(Ppambm) - GSL_IMAG(Ppapbm) - GSL_IMAG(Ppambp) );
                        
                        if (p == q) {
                            
                            InvQaRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                      - (C + M) - 2.0*(D2 + B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 + B1) * (2.0 - cos(Kperp)) - MixQaRe;
                            InvQaIm = 0.5*Gamma - MixQaIm;
                            
                            InvQbpRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                       - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 - B1) * (2.0 - cos(Kperp)) - MixQbpRe;
                            InvQbpIm = 0.5*Gamma - MixQbpIm;
                            
                            InvQbmRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                       - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 - B1) * (2.0 - cos(Kperp)) - MixQbmRe;
                            InvQbmIm = 0.5*Gamma - MixQbmIm;
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            InvQaRe = 0.25 + B1 - MixQaRe;
                            InvQaIm = - MixQaIm;
                            
                            InvQbpRe = 0.25 - B1 - MixQbpRe;
                            InvQbpIm = - MixQbpIm;
                            
                            InvQbmRe = 0.25 - B1 - MixQbmRe;
                            InvQbmIm = - MixQbmIm;
                            
                        } else {
                            
                            InvQaRe = - MixQaRe;
                            InvQaIm = - MixQaIm;
                            
                            InvQbpRe = - MixQbpRe;
                            InvQbpIm = - MixQbpIm;
                            
                            InvQbmRe = - MixQbmRe;
                            InvQbmIm = - MixQbmIm;
                        }
                        
                        gsl_matrix_complex_set (InvQa, p, q, gsl_complex_rect (InvQaRe, InvQaIm));
                        gsl_matrix_complex_set (InvQbp, p, q, gsl_complex_rect (InvQbpRe, InvQbpIm));
                        gsl_matrix_complex_set (InvQbm, p, q, gsl_complex_rect (InvQbmRe, InvQbmIm));
                    }
                }
                
                // Matrix inversion (2 bands)
                
                gsl_permutation *permQa = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Qa = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvQa, permQa, &perm);
                gsl_linalg_complex_LU_invert (InvQa, permQa, Qa);
                
                gsl_permutation *permQbp = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Qbp = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvQbp, permQbp, &perm);
                gsl_linalg_complex_LU_invert (InvQbp, permQbp, Qbp);
                
                gsl_permutation *permQbm = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Qbm = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvQbm, permQbm, &perm);
                gsl_linalg_complex_LU_invert (InvQbm, permQbm, Qbm);
                
                // Free the previous allocation
                
                gsl_permutation_free (permQa);
                gsl_matrix_complex_free (InvQa);
                
                gsl_permutation_free (permQbp);
                gsl_matrix_complex_free (InvQbp);
                
                gsl_permutation_free (permQbm);
                gsl_matrix_complex_free (InvQbm);
                
                
                /************************************************/
                /*** 3-band retarded lattice Green's function ***/
                /************************************************/
                
                gsl_matrix_complex *InvRap = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvRam = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvRbp = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvRbm = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        Ppab = gsl_matrix_complex_get (Pp, p, q);
                        Pmab = gsl_matrix_complex_get (Pm, p, q);
                        
                        if (p == 0) {
                            
                            Pmamb = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmamb = gsl_matrix_complex_get (Pm, p-1, q);
                        }
                        
                        if (p == Ncut-1) {
                            
                            Pmapb = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmapb = gsl_matrix_complex_get (Pm, p+1, q);
                        }
                        
                        if (q == 0) {
                            
                            Pmabm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmabm = gsl_matrix_complex_get (Pm, p, q-1);
                        }
                        
                        if (q == Ncut-1) {
                            
                            Pmabp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmabp = gsl_matrix_complex_get (Pm, p, q+1);
                        }
                        
                        if (p == 0 || q == 0) {
                            
                            Pmambm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmambm = gsl_matrix_complex_get (Pm, p-1, q-1);
                        }
                        
                        if (p == Ncut-1 || q == Ncut-1) {
                            
                            Pmapbp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmapbp = gsl_matrix_complex_get (Pm, p+1, q+1);
                        }
                        
                        if (p == 0 || q == Ncut-1) {
                            
                            Pmambp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmambp = gsl_matrix_complex_get (Pm, p-1, q+1);
                        }
                        
                        if (p == Ncut-1 || q == 0) {
                            
                            Pmapbm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmapbm = gsl_matrix_complex_get (Pm, p+1, q-1);
                        }
                        
                        MixRapRe = pow(0.5*A1,2) * ( pow(2.0*sin(Kperp),2) * GSL_REAL(Pmab)
                                   + 2.0*sin(Kperp) * ( GSL_REAL(Pmamb) - GSL_REAL(Pmapb) + GSL_REAL(Pmabm) - GSL_REAL(Pmabp) )
                                   + GSL_REAL(Pmambm) + GSL_REAL(Pmapbp) - GSL_REAL(Pmambp) - GSL_REAL(Pmapbm) );
                        MixRapIm = pow(0.5*A1,2) * ( pow(2.0*sin(Kperp),2) * GSL_IMAG(Pmab)
                                   + 2.0*sin(Kperp) * ( GSL_IMAG(Pmamb) - GSL_IMAG(Pmapb) + GSL_IMAG(Pmabm) - GSL_IMAG(Pmabp) )
                                   + GSL_IMAG(Pmambm) + GSL_IMAG(Pmapbp) - GSL_IMAG(Pmambp) - GSL_IMAG(Pmapbm) );
                        
                        MixRamRe = pow(0.5*A1,2) * ( pow(2.0*sin(Kperp),2) * GSL_REAL(Pmab)
                                   + 2.0*sin(Kperp) * ( GSL_REAL(Pmapb) - GSL_REAL(Pmamb) + GSL_REAL(Pmabp) - GSL_REAL(Pmabm) )
                                   + GSL_REAL(Pmapbp) + GSL_REAL(Pmambm) - GSL_REAL(Pmapbm) - GSL_REAL(Pmambp) );
                        MixRamIm = pow(0.5*A1,2) * ( pow(2.0*sin(Kperp),2) * GSL_IMAG(Pmab)
                                   + 2.0*sin(Kperp) * ( GSL_IMAG(Pmapb) - GSL_IMAG(Pmamb) + GSL_IMAG(Pmabp) - GSL_IMAG(Pmabm) )
                                   + GSL_IMAG(Pmapbp) + GSL_IMAG(Pmambm) - GSL_IMAG(Pmapbm) - GSL_IMAG(Pmambp) );
                        
                        MixRbpRe = MixRbmRe = pow(A2*sin(Kplane),2) * GSL_REAL(Ppab);
                        MixRbpIm = MixRbmIm = pow(A2*sin(Kplane),2) * GSL_IMAG(Ppab);
                        
                        for (r = 0; r <= Ncut-1; r++)
                        {
                            for (s = 0; s <= Ncut-1; s++)
                            {
                                PmLab = gsl_matrix_complex_get (Pm, p, r);
                                PmRab = gsl_matrix_complex_get (Pm, s, q);
                                
                                if (p == 0) {
                                    
                                    PmLamb = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PmLamb = gsl_matrix_complex_get (Pm, p-1, r);
                                }
                                
                                if (p == Ncut-1) {
                                    
                                    PmLapb = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PmLapb = gsl_matrix_complex_get (Pm, p+1, r);
                                }
                                
                                if (q == 0) {
                                    
                                    PmRabm = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PmRabm = gsl_matrix_complex_get (Pm, s, q-1);
                                }
                                
                                if (q == Ncut-1) {
                                    
                                    PmRabp = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PmRabp = gsl_matrix_complex_get (Pm, s, q+1);
                                }
 
                                PpLab = gsl_matrix_complex_get (Pp, p, r);
                                PpRab = gsl_matrix_complex_get (Pp, s, q);
                                
                                if (r == 0) {
                                    
                                    PpLabm = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PpLabm = gsl_matrix_complex_get (Pp, p, r-1);
                                }
                                
                                if (r == Ncut-1) {
                                    
                                    PpLabp = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PpLabp = gsl_matrix_complex_get (Pp, p, r+1);
                                }
                                
                                if (s == 0) {
                                    
                                    PpRamb = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PpRamb = gsl_matrix_complex_get (Pp, s-1, q);
                                }
                                
                                if (s == Ncut-1) {
                                    
                                    PpRapb = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PpRapb = gsl_matrix_complex_get (Pp, s+1, q);
                                }
                                
                                Qaab = gsl_matrix_complex_get (Qa, r, s);
                                Qbpab = gsl_matrix_complex_get (Qbp, r, s);
                                Qbmab = gsl_matrix_complex_get (Qbm, r, s);
                                
                                MixRapRe += pow(0.5*A1*A2*sin(Kplane),2)
                                            * ( ( ( 2.0*sin(Kperp) * GSL_REAL(PmLab) + GSL_REAL(PmLamb) - GSL_REAL(PmLapb) ) * GSL_REAL(Qaab)
                                            - ( 2.0*sin(Kperp) * GSL_IMAG(PmLab) + GSL_IMAG(PmLamb) - GSL_IMAG(PmLapb) ) * GSL_IMAG(Qaab) )
                                            * ( 2.0*sin(Kperp) * GSL_REAL(PmRab) + GSL_REAL(PmRabm) - GSL_REAL(PmRabp) )
                                            - ( ( 2.0*sin(Kperp) * GSL_REAL(PmLab) + GSL_REAL(PmLamb) - GSL_REAL(PmLapb) ) * GSL_IMAG(Qaab)
                                            + ( 2.0*sin(Kperp) * GSL_IMAG(PmLab) + GSL_IMAG(PmLamb) - GSL_IMAG(PmLapb) ) * GSL_REAL(Qaab) )
                                            * ( 2.0*sin(Kperp) * GSL_IMAG(PmRab) + GSL_IMAG(PmRabm) - GSL_IMAG(PmRabp) ) );
                                
                                MixRapIm += pow(0.5*A1*A2*sin(Kplane),2)
                                            * ( ( ( 2.0*sin(Kperp) * GSL_REAL(PmLab) + GSL_REAL(PmLamb) - GSL_REAL(PmLapb) ) * GSL_REAL(Qaab)
                                            - ( 2.0*sin(Kperp) * GSL_IMAG(PmLab) + GSL_IMAG(PmLamb) - GSL_IMAG(PmLapb) ) * GSL_IMAG(Qaab) )
                                            * ( 2.0*sin(Kperp) * GSL_IMAG(PmRab) + GSL_IMAG(PmRabm) - GSL_IMAG(PmRabp) )
                                            + ( ( 2.0*sin(Kperp) * GSL_REAL(PmLab) + GSL_REAL(PmLamb) - GSL_REAL(PmLapb) ) * GSL_IMAG(Qaab)
                                            + ( 2.0*sin(Kperp) * GSL_IMAG(PmLab) + GSL_IMAG(PmLamb) - GSL_IMAG(PmLapb) ) * GSL_REAL(Qaab) )
                                            * ( 2.0*sin(Kperp) * GSL_REAL(PmRab) + GSL_REAL(PmRabm) - GSL_REAL(PmRabp) ) );
                                
                                MixRamRe += pow(0.5*A1*A2*sin(Kplane),2)
                                            * ( ( ( 2.0*sin(Kperp) * GSL_REAL(PmLab) + GSL_REAL(PmLapb) - GSL_REAL(PmLamb) ) * GSL_REAL(Qaab)
                                            - ( 2.0*sin(Kperp) * GSL_IMAG(PmLab) + GSL_IMAG(PmLapb) - GSL_IMAG(PmLamb) ) * GSL_IMAG(Qaab) )
                                            * ( 2.0*sin(Kperp) * GSL_REAL(PmRab) + GSL_REAL(PmRabp) - GSL_REAL(PmRabm) )
                                            - ( ( 2.0*sin(Kperp) * GSL_REAL(PmLab) + GSL_REAL(PmLapb) - GSL_REAL(PmLamb) ) * GSL_IMAG(Qaab)
                                            + ( 2.0*sin(Kperp) * GSL_IMAG(PmLab) + GSL_IMAG(PmLapb) - GSL_IMAG(PmLamb) ) * GSL_REAL(Qaab) )
                                            * ( 2.0*sin(Kperp) * GSL_IMAG(PmRab) + GSL_IMAG(PmRabp) - GSL_IMAG(PmRabm) ) );
                                
                                MixRamIm += pow(0.5*A1*A2*sin(Kplane),2)
                                            * ( ( ( 2.0*sin(Kperp) * GSL_REAL(PmLab) + GSL_REAL(PmLapb) - GSL_REAL(PmLamb) ) * GSL_REAL(Qaab)
                                            - ( 2.0*sin(Kperp) * GSL_IMAG(PmLab) + GSL_IMAG(PmLapb) - GSL_IMAG(PmLamb) ) * GSL_IMAG(Qaab) )
                                            * ( 2.0*sin(Kperp) * GSL_IMAG(PmRab) + GSL_IMAG(PmRabp) - GSL_IMAG(PmRabm) )
                                            + ( ( 2.0*sin(Kperp) * GSL_REAL(PmLab) + GSL_REAL(PmLapb) - GSL_REAL(PmLamb) ) * GSL_IMAG(Qaab)
                                            + ( 2.0*sin(Kperp) * GSL_IMAG(PmLab) + GSL_IMAG(PmLapb) - GSL_IMAG(PmLamb) ) * GSL_REAL(Qaab) )
                                            * ( 2.0*sin(Kperp) * GSL_REAL(PmRab) + GSL_REAL(PmRabp) - GSL_REAL(PmRabm) ) );
                                
                                MixRbpRe += pow(0.5*A1*A2*sin(Kplane),2)
                                            * ( ( ( 2.0*sin(Kperp) * GSL_REAL(PpLab) + GSL_REAL(PpLabm) - GSL_REAL(PpLabp) ) * GSL_REAL(Qbpab)
                                            - ( 2.0*sin(Kperp) * GSL_IMAG(PpLab) + GSL_IMAG(PpLabm) - GSL_IMAG(PpLabp) ) * GSL_IMAG(Qbpab) )
                                            * ( 2.0*sin(Kperp) * GSL_REAL(PpRab) + GSL_REAL(PpRamb) - GSL_REAL(PpRapb) )
                                            - ( ( 2.0*sin(Kperp) * GSL_REAL(PpLab) + GSL_REAL(PpLabm) - GSL_REAL(PpLabp) ) * GSL_IMAG(Qbpab)
                                            + ( 2.0*sin(Kperp) * GSL_IMAG(PpLab) + GSL_IMAG(PpLabm) - GSL_IMAG(PpLabp) ) * GSL_REAL(Qbpab) )
                                            * ( 2.0*sin(Kperp) * GSL_IMAG(PpRab) + GSL_IMAG(PpRamb) - GSL_IMAG(PpRapb) ) );
                                
                                MixRbpIm += pow(0.5*A1*A2*sin(Kplane),2)
                                            * ( ( ( 2.0*sin(Kperp) * GSL_REAL(PpLab) + GSL_REAL(PpLabm) - GSL_REAL(PpLabp) ) * GSL_REAL(Qbpab)
                                            - ( 2.0*sin(Kperp) * GSL_IMAG(PpLab) + GSL_IMAG(PpLabm) - GSL_IMAG(PpLabp) ) * GSL_IMAG(Qbpab) )
                                            * ( 2.0*sin(Kperp) * GSL_IMAG(PpRab) + GSL_IMAG(PpRamb) - GSL_IMAG(PpRapb) )
                                            + ( ( 2.0*sin(Kperp) * GSL_REAL(PpLab) + GSL_REAL(PpLabm) - GSL_REAL(PpLabp) ) * GSL_IMAG(Qbpab)
                                            + ( 2.0*sin(Kperp) * GSL_IMAG(PpLab) + GSL_IMAG(PpLabm) - GSL_IMAG(PpLabp) ) * GSL_REAL(Qbpab) )
                                            * ( 2.0*sin(Kperp) * GSL_REAL(PpRab) + GSL_REAL(PpRamb) - GSL_REAL(PpRapb) ) );
                                
                                MixRbmRe += pow(0.5*A1*A2*sin(Kplane),2)
                                            * ( ( ( 2.0*sin(Kperp) * GSL_REAL(PpLab) + GSL_REAL(PpLabp) - GSL_REAL(PpLabm) ) * GSL_REAL(Qbmab)
                                            - ( 2.0*sin(Kperp) * GSL_IMAG(PpLab) + GSL_IMAG(PpLabp) - GSL_IMAG(PpLabm) ) * GSL_IMAG(Qbmab) )
                                            * ( 2.0*sin(Kperp) * GSL_REAL(PpRab) + GSL_REAL(PpRapb) - GSL_REAL(PpRamb) )
                                            - ( ( 2.0*sin(Kperp) * GSL_REAL(PpLab) + GSL_REAL(PpLabp) - GSL_REAL(PpLabm) ) * GSL_IMAG(Qbmab)
                                            + ( 2.0*sin(Kperp) * GSL_IMAG(PpLab) + GSL_IMAG(PpLabp) - GSL_IMAG(PpLabm) ) * GSL_REAL(Qbmab) )
                                            * ( 2.0*sin(Kperp) * GSL_IMAG(PpRab) + GSL_IMAG(PpRapb) - GSL_IMAG(PpRamb) ) );
                                
                                MixRbmIm += pow(0.5*A1*A2*sin(Kplane),2)
                                            * ( ( ( 2.0*sin(Kperp) * GSL_REAL(PpLab) + GSL_REAL(PpLabp) - GSL_REAL(PpLabm) ) * GSL_REAL(Qbmab)
                                            - ( 2.0*sin(Kperp) * GSL_IMAG(PpLab) + GSL_IMAG(PpLabp) - GSL_IMAG(PpLabm) ) * GSL_IMAG(Qbmab) )
                                            * ( 2.0*sin(Kperp) * GSL_IMAG(PpRab) + GSL_IMAG(PpRapb) - GSL_IMAG(PpRamb) )
                                            + ( ( 2.0*sin(Kperp) * GSL_REAL(PpLab) + GSL_REAL(PpLabp) - GSL_REAL(PpLabm) ) * GSL_IMAG(Qbmab)
                                            + ( 2.0*sin(Kperp) * GSL_IMAG(PpLab) + GSL_IMAG(PpLabp) - GSL_IMAG(PpLabm) ) * GSL_REAL(Qbmab) )
                                            * ( 2.0*sin(Kperp) * GSL_REAL(PpRab) + GSL_REAL(PpRapb) - GSL_REAL(PpRamb) ) );
                            }
                        }
                        
                        if (p == q) {
                            
                            InvRapRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                       - (C + M) - 2.0*(D2 + B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 + B1) * (2.0 - cos(Kperp)) - MixRapRe;
                            InvRapIm = 0.5*Gamma - MixRapIm;
                            
                            InvRamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                       - (C + M) - 2.0*(D2 + B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 + B1) * (2.0 - cos(Kperp)) - MixRamRe;
                            InvRamIm = 0.5*Gamma - MixRamIm;
                            
                            InvRbpRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                       - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 - B1) * (2.0 - cos(Kperp)) - MixRbpRe;
                            InvRbpIm = 0.5*Gamma - MixRbpIm;
                            
                            InvRbmRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                       - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 - B1) * (2.0 - cos(Kperp)) - MixRbmRe;
                            InvRbmIm = 0.5*Gamma - MixRbmIm;
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            InvRapRe = 0.25 + B1 - MixRapRe;
                            InvRapIm = - MixRapIm;
                            
                            InvRamRe = 0.25 + B1 - MixRamRe;
                            InvRamIm = - MixRamIm;
                            
                            InvRbpRe = 0.25 - B1 - MixRbpRe;
                            InvRbpIm = - MixRbpIm;
                            
                            InvRbmRe = 0.25 - B1 - MixRbmRe;
                            InvRbmIm = - MixRbmIm;
                            
                        } else {
                            
                            InvRapRe = - MixRapRe;
                            InvRapIm = - MixRapIm;
                            
                            InvRamRe = - MixRamRe;
                            InvRamIm = - MixRamIm;
                            
                            InvRbpRe = - MixRbpRe;
                            InvRbpIm = - MixRbpIm;
                            
                            InvRbmRe = - MixRbmRe;
                            InvRbmIm = - MixRbmIm;
                        }
                        
                        gsl_matrix_complex_set (InvRap, p, q, gsl_complex_rect (InvRapRe, InvRapIm));
                        gsl_matrix_complex_set (InvRam, p, q, gsl_complex_rect (InvRamRe, InvRamIm));
                        gsl_matrix_complex_set (InvRbp, p, q, gsl_complex_rect (InvRbpRe, InvRbpIm));
                        gsl_matrix_complex_set (InvRbm, p, q, gsl_complex_rect (InvRbmRe, InvRbmIm));
                    }
                }
                
                // Matrix inversion (3 bands)
                
                gsl_permutation *permRap = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Rap = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvRap, permRap, &perm);
                gsl_linalg_complex_LU_invert (InvRap, permRap, Rap);
                
                gsl_permutation *permRam = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Ram = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvRam, permRam, &perm);
                gsl_linalg_complex_LU_invert (InvRam, permRam, Ram);
                
                gsl_permutation *permRbp = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Rbp = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvRbp, permRbp, &perm);
                gsl_linalg_complex_LU_invert (InvRbp, permRbp, Rbp);
                
                gsl_permutation *permRbm = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Rbm = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvRbm, permRbm, &perm);
                gsl_linalg_complex_LU_invert (InvRbm, permRbm, Rbm);
                
                // Free the previous allocation
                
                gsl_permutation_free (permRap);
                gsl_matrix_complex_free (InvRap);
                
                gsl_permutation_free (permRam);
                gsl_matrix_complex_free (InvRam);
                
                gsl_permutation_free (permRbp);
                gsl_matrix_complex_free (InvRbp);
                
                gsl_permutation_free (permRbm);
                gsl_matrix_complex_free (InvRbm);
                
                
                /************************************************/
                /*** 4-band retarded lattice Green's function ***/
                /************************************************/
                
                gsl_matrix_complex *MaL = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *MaR = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *MbL = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *MbR = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *McL = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *McR = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *MdL = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *MdR = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                gsl_matrix_complex *InvGa = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvGb = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvGc = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvGd = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        if (p == q) {
                            
                            MaLRe = MaRRe = A2*sin(Kplane);
                            MaLIm = MaRIm = 0;
                            
                            McLRe = McRRe = A2*sin(Kplane);
                            McLIm = McRIm = 0;
                            
                        } else {
                            
                            MaLRe = MaRRe = 0;
                            MaLIm = MaRIm = 0;
                            
                            McLRe = McRRe = 0;
                            McLIm = McRIm = 0;
                        }
                        
                        if (p == q) {
                            
                            MbLRe = MbRRe = A1*sin(Kperp);
                            MbLIm = MbRIm = 0;
                            
                            MdLRe = MdRRe = A1*sin(Kperp);
                            MdLIm = MdRIm = 0;
                        
                        } else if (p == q + 1 || p == q - 1) {
                            
                            MbLRe = (p-q)*0.5*A1;
                            MbRRe = -(p-q)*0.5*A1;
                            MbLIm = MbRIm = 0;
                            
                            MdLRe = -(p-q)*0.5*A1;
                            MdRRe = (p-q)*0.5*A1;
                            MdLIm = MdRIm = 0;
                            
                        } else {
                            
                            MbLRe = MbRRe = 0;
                            MbLIm = MbRIm = 0;
                            
                            MdLRe = MdRRe = 0;
                            MdLIm = MdRIm = 0;
                        }
                        
                        for (r = 0; r <= Ncut-1; r++)
                        {
                            QaLab = gsl_matrix_complex_get (Qa, p, r);
                            QaRab = gsl_matrix_complex_get (Qa, r, q);
                            
                            QbpLab = gsl_matrix_complex_get (Qbp, p, r);
                            QbpRab = gsl_matrix_complex_get (Qbp, r, q);
                            
                            QbmLab = gsl_matrix_complex_get (Qbm, p, r);
                            QbmRab = gsl_matrix_complex_get (Qbm, r, q);
                            
                            PpLab = gsl_matrix_complex_get (Pp, p, r);
                            PpRab = gsl_matrix_complex_get (Pp, r, q);
                            
                            PmLab = gsl_matrix_complex_get (Pm, p, r);
                            PmRab = gsl_matrix_complex_get (Pm, r, q);
                            
                            if (p == 0) {
                                
                                Qaamb = gsl_complex_rect (0, 0);
                                Pmamb = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                Qaamb = gsl_matrix_complex_get (Qa, p-1, r);
                                Pmamb = gsl_matrix_complex_get (Pm, p-1, r);
                            }
                            
                            if (p == Ncut-1) {
                                
                                Qaapb = gsl_complex_rect (0, 0);
                                Pmapb = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                Qaapb = gsl_matrix_complex_get (Qa, p+1, r);
                                Pmapb = gsl_matrix_complex_get (Pm, p+1, r);
                            }
                            
                            if (q == 0) {
                                
                                Qaabm = gsl_complex_rect (0, 0);
                                Pmabm = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                Qaabm = gsl_matrix_complex_get (Qa, r, q-1);
                                Pmabm = gsl_matrix_complex_get (Pm, r, q-1);
                            }
                            
                            if (q == Ncut-1) {
                                
                                Qaabp = gsl_complex_rect (0, 0);
                                Pmabp = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                Qaabp = gsl_matrix_complex_get (Qa, r, q+1);
                                Pmabp = gsl_matrix_complex_get (Pm, r, q+1);
                            }
                            
                            if (r == 0) {
                                
                                Qbpabm = gsl_complex_rect (0, 0);
                                Qbmabm = gsl_complex_rect (0, 0);
                                Ppabm = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                Qbpabm = gsl_matrix_complex_get (Qbp, p, r-1);
                                Qbmabm = gsl_matrix_complex_get (Qbm, p, r-1);
                                Ppabm = gsl_matrix_complex_get (Pp, p, r-1);
                            }
                            
                            if (r == Ncut-1) {
                                
                                Qbpabp = gsl_complex_rect (0, 0);
                                Qbmabp = gsl_complex_rect (0, 0);
                                Ppabp = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                Qbpabp = gsl_matrix_complex_get (Qbp, p, r+1);
                                Qbmabp = gsl_matrix_complex_get (Qbm, p, r+1);
                                Ppabp = gsl_matrix_complex_get (Pp, p, r+1);
                            }
                            
                            MaLRe += - pow(0.5*A1,2) * A2*sin(Kplane)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(QaLab) + GSL_REAL(Qaamb) - GSL_REAL(Qaapb) )
                                     * ( 2.0*sin(Kperp) * GSL_REAL(PmRab) + GSL_REAL(Pmabm) - GSL_REAL(Pmabp) )
                                     - ( 2.0*sin(Kperp) * GSL_IMAG(QaLab) + GSL_IMAG(Qaamb) - GSL_IMAG(Qaapb) )
                                     * ( 2.0*sin(Kperp) * GSL_IMAG(PmRab) + GSL_IMAG(Pmabm) - GSL_IMAG(Pmabp) ) );
                            MaLIm += - pow(0.5*A1,2) * A2*sin(Kplane)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(QaLab) + GSL_REAL(Qaamb) - GSL_REAL(Qaapb) )
                                     * ( 2.0*sin(Kperp) * GSL_IMAG(PmRab) + GSL_IMAG(Pmabm) - GSL_IMAG(Pmabp) )
                                     + ( 2.0*sin(Kperp) * GSL_IMAG(QaLab) + GSL_IMAG(Qaamb) - GSL_IMAG(Qaapb) )
                                     * ( 2.0*sin(Kperp) * GSL_REAL(PmRab) + GSL_REAL(Pmabm) - GSL_REAL(Pmabp) ) );
                            
                            MaRRe += - pow(0.5*A1,2) * A2*sin(Kplane)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(PmLab) + GSL_REAL(Pmamb) - GSL_REAL(Pmapb) )
                                     * ( 2.0*sin(Kperp) * GSL_REAL(QaRab) + GSL_REAL(Qaabm) - GSL_REAL(Qaabp) )
                                     - ( 2.0*sin(Kperp) * GSL_IMAG(PmLab) + GSL_IMAG(Pmamb) - GSL_IMAG(Pmapb) )
                                     * ( 2.0*sin(Kperp) * GSL_IMAG(QaRab) + GSL_IMAG(Qaabm) - GSL_IMAG(Qaabp) ) );
                            MaRIm += - pow(0.5*A1,2) * A2*sin(Kplane)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(PmLab) + GSL_REAL(Pmamb) - GSL_REAL(Pmapb) )
                                     * ( 2.0*sin(Kperp) * GSL_IMAG(QaRab) + GSL_IMAG(Qaabm) - GSL_IMAG(Qaabp) )
                                     + ( 2.0*sin(Kperp) * GSL_IMAG(PmLab) + GSL_IMAG(Pmamb) - GSL_IMAG(Pmapb) )
                                     * ( 2.0*sin(Kperp) * GSL_REAL(QaRab) + GSL_REAL(Qaabm) - GSL_REAL(Qaabp) ) );
                            
                            MbLRe += - 0.5*A1 * pow(A2*sin(Kplane),2)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(QbpLab) + GSL_REAL(Qbpabp) - GSL_REAL(Qbpabm) ) * GSL_REAL(PpRab)
                                     - ( 2.0*sin(Kperp) * GSL_IMAG(QbpLab) + GSL_IMAG(Qbpabp) - GSL_IMAG(Qbpabm) ) * GSL_IMAG(PpRab) );
                            MbLIm += - 0.5*A1 * pow(A2*sin(Kplane),2)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(QbpLab) + GSL_REAL(Qbpabp) - GSL_REAL(Qbpabm) ) * GSL_IMAG(PpRab)
                                     + ( 2.0*sin(Kperp) * GSL_IMAG(QbpLab) + GSL_IMAG(Qbpabp) - GSL_IMAG(Qbpabm) ) * GSL_REAL(PpRab) );
                            
                            MbRRe += - 0.5*A1 * pow(A2*sin(Kplane),2)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(PpLab) + GSL_REAL(Ppabm) - GSL_REAL(Ppabp) ) * GSL_REAL(QbpRab)
                                     - ( 2.0*sin(Kperp) * GSL_IMAG(PpLab) + GSL_IMAG(Ppabm) - GSL_IMAG(Ppabp) ) * GSL_IMAG(QbpRab) );
                            MbRIm += - 0.5*A1 * pow(A2*sin(Kplane),2)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(PpLab) + GSL_REAL(Ppabm) - GSL_REAL(Ppabp) ) * GSL_IMAG(QbpRab)
                                     + ( 2.0*sin(Kperp) * GSL_IMAG(PpLab) + GSL_IMAG(Ppabm) - GSL_IMAG(Ppabp) ) * GSL_REAL(QbpRab) );
                            
                            McLRe += - pow(0.5*A1,2) * A2*sin(Kplane)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(QaLab) + GSL_REAL(Qaapb) - GSL_REAL(Qaamb) )
                                     * ( 2.0*sin(Kperp) * GSL_REAL(PmRab) + GSL_REAL(Pmabp) - GSL_REAL(Pmabm) )
                                     - ( 2.0*sin(Kperp) * GSL_IMAG(QaLab) + GSL_IMAG(Qaapb) - GSL_IMAG(Qaamb) )
                                     * ( 2.0*sin(Kperp) * GSL_IMAG(PmRab) + GSL_IMAG(Pmabp) - GSL_IMAG(Pmabm) ) );
                            McLIm += - pow(0.5*A1,2) * A2*sin(Kplane)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(QaLab) + GSL_REAL(Qaapb) - GSL_REAL(Qaamb) )
                                     * ( 2.0*sin(Kperp) * GSL_IMAG(PmRab) + GSL_IMAG(Pmabp) - GSL_IMAG(Pmabm) )
                                     + ( 2.0*sin(Kperp) * GSL_IMAG(QaLab) + GSL_IMAG(Qaapb) - GSL_IMAG(Qaamb) )
                                     * ( 2.0*sin(Kperp) * GSL_REAL(PmRab) + GSL_REAL(Pmabp) - GSL_REAL(Pmabm) ) );
                            
                            McRRe += - pow(0.5*A1,2) * A2*sin(Kplane)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(PmLab) + GSL_REAL(Pmapb) - GSL_REAL(Pmamb) )
                                     * ( 2.0*sin(Kperp) * GSL_REAL(QaRab) + GSL_REAL(Qaabp) - GSL_REAL(Qaabm) )
                                     - ( 2.0*sin(Kperp) * GSL_IMAG(PmLab) + GSL_IMAG(Pmapb) - GSL_IMAG(Pmamb) )
                                     * ( 2.0*sin(Kperp) * GSL_IMAG(QaRab) + GSL_IMAG(Qaabp) - GSL_IMAG(Qaabm) ) );
                            McRIm += - pow(0.5*A1,2) * A2*sin(Kplane)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(PmLab) + GSL_REAL(Pmapb) - GSL_REAL(Pmamb) )
                                     * ( 2.0*sin(Kperp) * GSL_IMAG(QaRab) + GSL_IMAG(Qaabp) - GSL_IMAG(Qaabm) )
                                     + ( 2.0*sin(Kperp) * GSL_IMAG(PmLab) + GSL_IMAG(Pmapb) - GSL_IMAG(Pmamb) )
                                     * ( 2.0*sin(Kperp) * GSL_REAL(QaRab) + GSL_REAL(Qaabp) - GSL_REAL(Qaabm) ) );
                            
                            MdLRe += - 0.5*A1 * pow(A2*sin(Kplane),2)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(QbmLab) + GSL_REAL(Qbmabm) - GSL_REAL(Qbmabp) ) * GSL_REAL(PpRab)
                                     - ( 2.0*sin(Kperp) * GSL_IMAG(QbmLab) + GSL_IMAG(Qbmabm) - GSL_IMAG(Qbmabp) ) * GSL_IMAG(PpRab) );
                            MdLIm += - 0.5*A1 * pow(A2*sin(Kplane),2)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(QbmLab) + GSL_REAL(Qbmabm) - GSL_REAL(Qbmabp) ) * GSL_IMAG(PpRab)
                                     + ( 2.0*sin(Kperp) * GSL_IMAG(QbmLab) + GSL_IMAG(Qbmabm) - GSL_IMAG(Qbmabp) ) * GSL_REAL(PpRab) );
                            
                            MdRRe += - 0.5*A1 * pow(A2*sin(Kplane),2)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(PpLab) + GSL_REAL(Ppabp) - GSL_REAL(Ppabm) ) * GSL_REAL(QbmRab)
                                     - ( 2.0*sin(Kperp) * GSL_IMAG(PpLab) + GSL_IMAG(Ppabp) - GSL_IMAG(Ppabm) ) * GSL_IMAG(QbmRab) );
                            MdRIm += - 0.5*A1 * pow(A2*sin(Kplane),2)
                                     * ( ( 2.0*sin(Kperp) * GSL_REAL(PpLab) + GSL_REAL(Ppabp) - GSL_REAL(Ppabm) ) * GSL_IMAG(QbmRab)
                                     + ( 2.0*sin(Kperp) * GSL_IMAG(PpLab) + GSL_IMAG(Ppabp) - GSL_IMAG(Ppabm) ) * GSL_REAL(QbmRab) );
                        }
                        
                        gsl_matrix_complex_set (MaL, p, q, gsl_complex_rect (MaLRe, MaLIm));
                        gsl_matrix_complex_set (MaR, p, q, gsl_complex_rect (MaRRe, MaRIm));
                        gsl_matrix_complex_set (MbL, p, q, gsl_complex_rect (MbLRe, MbLIm));
                        gsl_matrix_complex_set (MbR, p, q, gsl_complex_rect (MbRRe, MbRIm));
                        gsl_matrix_complex_set (McL, p, q, gsl_complex_rect (McLRe, McLIm));
                        gsl_matrix_complex_set (McR, p, q, gsl_complex_rect (McRRe, McRIm));
                        gsl_matrix_complex_set (MdL, p, q, gsl_complex_rect (MdLRe, MdLIm));
                        gsl_matrix_complex_set (MdR, p, q, gsl_complex_rect (MdRRe, MdRIm));
                    }
                }
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        Qaab = gsl_matrix_complex_get (Qa, p, q);
                        Qbpab = gsl_matrix_complex_get (Qbp, p, q);
                        Qbmab = gsl_matrix_complex_get (Qbm, p, q);
                        
                        if (p == 0) {
                            
                            Qaamb = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaamb = gsl_matrix_complex_get (Qa, p-1, q);
                        }
                        
                        if (p == Ncut-1) {
                            
                            Qaapb = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaapb = gsl_matrix_complex_get (Qa, p+1, q);
                        }
                        
                        if (q == 0) {
                            
                            Qaabm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaabm = gsl_matrix_complex_get (Qa, p, q-1);
                        }
                        
                        if (q == Ncut-1) {
                            
                            Qaabp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaabp = gsl_matrix_complex_get (Qa, p, q+1);
                        }
                        
                        if (p == 0 || q == 0) {
                            
                            Qaambm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaambm = gsl_matrix_complex_get (Qa, p-1, q-1);
                        }
                        
                        if (p == Ncut-1 || q == Ncut-1) {
                            
                            Qaapbp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaapbp = gsl_matrix_complex_get (Qa, p+1, q+1);
                        }
                        
                        if (p == 0 || q == Ncut-1) {
                            
                            Qaambp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaambp = gsl_matrix_complex_get (Qa, p-1, q+1);
                        }
                        
                        if (p == Ncut-1 || q == 0) {
                            
                            Qaapbm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaapbm = gsl_matrix_complex_get (Qa, p+1, q-1);
                        }
                        
                        MixGaRe = pow(0.5*A1,2) * ( pow(2.0*sin(Kperp),2) * GSL_REAL(Qaab)
                                  + 2.0*sin(Kperp) * ( GSL_REAL(Qaamb) - GSL_REAL(Qaapb) + GSL_REAL(Qaabm) - GSL_REAL(Qaabp) )
                                  + GSL_REAL(Qaambm) + GSL_REAL(Qaapbp) - GSL_REAL(Qaambp) - GSL_REAL(Qaapbm) );
                        MixGaIm = pow(0.5*A1,2) * ( pow(2.0*sin(Kperp),2) * GSL_IMAG(Qaab)
                                  + 2.0*sin(Kperp) * ( GSL_IMAG(Qaamb) - GSL_IMAG(Qaapb) + GSL_IMAG(Qaabm) - GSL_IMAG(Qaabp) )
                                  + GSL_IMAG(Qaambm) + GSL_IMAG(Qaapbp) - GSL_IMAG(Qaambp) - GSL_IMAG(Qaapbm) );
                        
                        MixGbRe = pow(A2*sin(Kplane),2) * GSL_REAL(Qbpab);
                        MixGbIm = pow(A2*sin(Kplane),2) * GSL_IMAG(Qbpab);
                        
                        MixGcRe = pow(0.5*A1,2) * ( pow(2.0*sin(Kperp),2) * GSL_REAL(Qaab)
                                  + 2.0*sin(Kperp) * ( GSL_REAL(Qaapb) - GSL_REAL(Qaamb) + GSL_REAL(Qaabp) - GSL_REAL(Qaabm) )
                                  + GSL_REAL(Qaapbp) + GSL_REAL(Qaambm) - GSL_REAL(Qaapbm) - GSL_REAL(Qaambp) );
                        MixGcIm = pow(0.5*A1,2) * ( pow(2.0*sin(Kperp),2) * GSL_IMAG(Qaab)
                                  + 2.0*sin(Kperp) * ( GSL_IMAG(Qaapb) - GSL_IMAG(Qaamb) + GSL_IMAG(Qaabp) - GSL_IMAG(Qaabm) )
                                  + GSL_IMAG(Qaapbp) + GSL_IMAG(Qaambm) - GSL_IMAG(Qaapbm) - GSL_IMAG(Qaambp) );
                        
                        MixGdRe = pow(A2*sin(Kplane),2) * GSL_REAL(Qbmab);
                        MixGdIm = pow(A2*sin(Kplane),2) * GSL_IMAG(Qbmab);
                        
                        for (r = 0; r <= Ncut-1; r++)
                        {
                            for (s = 0; s <= Ncut-1; s++)
                            {
                                MaLab = gsl_matrix_complex_get (MaL, p, r);
                                Rapab = gsl_matrix_complex_get (Rap, r, s);
                                MaRab = gsl_matrix_complex_get (MaR, s, q);
                                
                                MixGaRe += ( GSL_REAL(MaLab) * GSL_REAL(Rapab) - GSL_IMAG(MaLab) * GSL_IMAG(Rapab) ) * GSL_REAL(MaRab)
                                           - ( GSL_REAL(MaLab) * GSL_IMAG(Rapab) + GSL_IMAG(MaLab) * GSL_REAL(Rapab) ) * GSL_IMAG(MaRab);
                                
                                MixGaIm += ( GSL_REAL(MaLab) * GSL_REAL(Rapab) - GSL_IMAG(MaLab) * GSL_IMAG(Rapab) ) * GSL_IMAG(MaRab)
                                           + ( GSL_REAL(MaLab) * GSL_IMAG(Rapab) + GSL_IMAG(MaLab) * GSL_REAL(Rapab) ) * GSL_REAL(MaRab);
                                
                                MbLab = gsl_matrix_complex_get (MbL, p, r);
                                Rbpab = gsl_matrix_complex_get (Rbp, r, s);
                                MbRab = gsl_matrix_complex_get (MbR, s, q);
                                
                                MixGbRe += ( GSL_REAL(MbLab) * GSL_REAL(Rbpab) - GSL_IMAG(MbLab) * GSL_IMAG(Rbpab) ) * GSL_REAL(MbRab)
                                           - ( GSL_REAL(MbLab) * GSL_IMAG(Rbpab) + GSL_IMAG(MbLab) * GSL_REAL(Rbpab) ) * GSL_IMAG(MbRab);
                                
                                MixGbIm += ( GSL_REAL(MbLab) * GSL_REAL(Rbpab) - GSL_IMAG(MbLab) * GSL_IMAG(Rbpab) ) * GSL_IMAG(MbRab)
                                           + ( GSL_REAL(MbLab) * GSL_IMAG(Rbpab) + GSL_IMAG(MbLab) * GSL_REAL(Rbpab) ) * GSL_REAL(MbRab);
                                
                                McLab = gsl_matrix_complex_get (McL, p, r);
                                Ramab = gsl_matrix_complex_get (Ram, r, s);
                                McRab = gsl_matrix_complex_get (McR, s, q);
                                
                                MixGcRe += ( GSL_REAL(McLab) * GSL_REAL(Ramab) - GSL_IMAG(McLab) * GSL_IMAG(Ramab) ) * GSL_REAL(McRab)
                                           - ( GSL_REAL(McLab) * GSL_IMAG(Ramab) + GSL_IMAG(McLab) * GSL_REAL(Ramab) ) * GSL_IMAG(McRab);
                                
                                MixGcIm += ( GSL_REAL(McLab) * GSL_REAL(Ramab) - GSL_IMAG(McLab) * GSL_IMAG(Ramab) ) * GSL_IMAG(McRab)
                                           + ( GSL_REAL(McLab) * GSL_IMAG(Ramab) + GSL_IMAG(McLab) * GSL_REAL(Ramab) ) * GSL_REAL(McRab);
                                
                                MdLab = gsl_matrix_complex_get (MdL, p, r);
                                Rbmab = gsl_matrix_complex_get (Rbm, r, s);
                                MdRab = gsl_matrix_complex_get (MdR, s, q);
                                
                                MixGdRe += ( GSL_REAL(MdLab) * GSL_REAL(Rbmab) - GSL_IMAG(MdLab) * GSL_IMAG(Rbmab) ) * GSL_REAL(MdRab)
                                           - ( GSL_REAL(MdLab) * GSL_IMAG(Rbmab) + GSL_IMAG(MdLab) * GSL_REAL(Rbmab) ) * GSL_IMAG(MdRab);
                                
                                MixGdIm += ( GSL_REAL(MdLab) * GSL_REAL(Rbmab) - GSL_IMAG(MdLab) * GSL_IMAG(Rbmab) ) * GSL_IMAG(MdRab)
                                           + ( GSL_REAL(MdLab) * GSL_IMAG(Rbmab) + GSL_IMAG(MdLab) * GSL_REAL(Rbmab) ) * GSL_REAL(MdRab);
                            }
                        }
                        
                        if (p == q) {
                            
                            InvGaRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                      - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 - B1) * (2.0 - cos(Kperp)) - MixGaRe;
                            InvGaIm = 0.5*Gamma - MixGaIm;
                            
                            InvGbRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                      - (C + M) - 2.0*(D2 + B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 + B1) * (2.0 - cos(Kperp)) - MixGbRe;
                            InvGbIm = 0.5*Gamma - MixGbIm;
                            
                            InvGcRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                      - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 - B1) * (2.0 - cos(Kperp)) - MixGcRe;
                            InvGcIm = 0.5*Gamma - MixGcIm;
                            
                            InvGdRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                      - (C + M) - 2.0*(D2 + B2) * (1.0 - cos(Kplane)) - 2.0*(0.25 + B1) * (2.0 - cos(Kperp)) - MixGdRe;
                            InvGdIm = 0.5*Gamma - MixGdIm;
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            InvGaRe = 0.25 - B1 - MixGaRe;
                            InvGaIm = - MixGaIm;
                            
                            InvGbRe = 0.25 + B1 - MixGbRe;
                            InvGbIm = - MixGbIm;
                            
                            InvGcRe = 0.25 - B1 - MixGcRe;
                            InvGcIm = - MixGcIm;
                            
                            InvGdRe = 0.25 + B1 - MixGdRe;
                            InvGdIm = - MixGdIm;
                            
                        } else {
                            
                            InvGaRe = - MixGaRe;
                            InvGaIm = - MixGaIm;
                            
                            InvGbRe = - MixGbRe;
                            InvGbIm = - MixGbIm;
                            
                            InvGcRe = - MixGcRe;
                            InvGcIm = - MixGcIm;
                            
                            InvGdRe = - MixGdRe;
                            InvGdIm = - MixGdIm;
                        }
                        
                        gsl_matrix_complex_set (InvGa, p, q, gsl_complex_rect (InvGaRe, InvGaIm));
                        gsl_matrix_complex_set (InvGb, p, q, gsl_complex_rect (InvGbRe, InvGbIm));
                        gsl_matrix_complex_set (InvGc, p, q, gsl_complex_rect (InvGcRe, InvGcIm));
                        gsl_matrix_complex_set (InvGd, p, q, gsl_complex_rect (InvGdRe, InvGdIm));
                    }
                }
                
                // Free the previous allocation
                
                gsl_matrix_complex_free (Pp);
                gsl_matrix_complex_free (Pm);
                
                gsl_matrix_complex_free (Qa);
                gsl_matrix_complex_free (Qbp);
                gsl_matrix_complex_free (Qbm);
                
                gsl_matrix_complex_free (Rap);
                gsl_matrix_complex_free (Ram);
                gsl_matrix_complex_free (Rbp);
                gsl_matrix_complex_free (Rbm);
                
                gsl_matrix_complex_free (MaL);
                gsl_matrix_complex_free (MaR);
                gsl_matrix_complex_free (MbL);
                gsl_matrix_complex_free (MbR);
                gsl_matrix_complex_free (McL);
                gsl_matrix_complex_free (McR);
                gsl_matrix_complex_free (MdL);
                gsl_matrix_complex_free (MdR);
                
                // Matrix inversion (4 bands)
                
                gsl_permutation *permGa = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Ga = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvGa, permGa, &perm);
                gsl_linalg_complex_LU_invert (InvGa, permGa, Ga);
                
                gsl_permutation *permGb = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Gb = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvGb, permGb, &perm);
                gsl_linalg_complex_LU_invert (InvGb, permGb, Gb);
                
                gsl_permutation *permGc = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Gc = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvGc, permGc, &perm);
                gsl_linalg_complex_LU_invert (InvGc, permGc, Gc);
                
                gsl_permutation *permGd = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Gd = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvGd, permGd, &perm);
                gsl_linalg_complex_LU_invert (InvGd, permGd, Gd);
                
                // 4-band retarded local Green's function
                
                gsl_complex kGa = gsl_matrix_complex_get (Ga, n, n);
                gsl_complex kGb = gsl_matrix_complex_get (Gb, n, n);
                gsl_complex kGc = gsl_matrix_complex_get (Gc, n, n);
                gsl_complex kGd = gsl_matrix_complex_get (Gd, n, n);
                
                // Semi-local Dos
                
                OrbA(SemiLocDosSub,iw+Fmax*(n-rank)/Ncpu) = - GSL_IMAG(kGa) / M_PI;
                OrbB(SemiLocDosSub,iw+Fmax*(n-rank)/Ncpu) = - GSL_IMAG(kGb) / M_PI;
                OrbC(SemiLocDosSub,iw+Fmax*(n-rank)/Ncpu) = - GSL_IMAG(kGc) / M_PI;
                OrbD(SemiLocDosSub,iw+Fmax*(n-rank)/Ncpu) = - GSL_IMAG(kGd) / M_PI;
                
                // Free the previous allocation
                
                gsl_permutation_free (permGa);
                gsl_matrix_complex_free (InvGa);
                gsl_matrix_complex_free (Ga);
                
                gsl_permutation_free (permGb);
                gsl_matrix_complex_free (InvGb);
                gsl_matrix_complex_free (Gb);
                
                gsl_permutation_free (permGc);
                gsl_matrix_complex_free (InvGc);
                gsl_matrix_complex_free (Gc);
                
                gsl_permutation_free (permGd);
                gsl_matrix_complex_free (InvGd);
                gsl_matrix_complex_free (Gd);
            }
            
            
            /**************************************/
            /******* yz subsystems under Ey *******/
            /**************************************/
            
            if (Mod == 2) {
            
                /************************************************/
                /*** 1-band retarded lattice Green's function ***/
                /************************************************/
                
                gsl_matrix_complex *InvPp = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvPm = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        if (p == q) {
                            
                            InvPpRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                    - (C + M) - 2.0*(D2 + B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 + B1) * (2.0 - cos(Kplane));
                            InvPpIm = 0.5*Gamma;
                            
                            InvPmRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                    - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 - B1) * (2.0 - cos(Kplane));
                            InvPmIm = 0.5*Gamma;
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            InvPpRe = 0.25 + B1;
                            InvPpIm = 0;
                            
                            InvPmRe = 0.25 - B1;
                            InvPmIm = 0;
                            
                        } else {
                            
                            InvPpRe = InvPpIm = 0;
                            InvPmRe = InvPmIm = 0;
                        }
                        
                        gsl_matrix_complex_set (InvPp, p, q, gsl_complex_rect (InvPpRe, InvPpIm));
                        gsl_matrix_complex_set (InvPm, p, q, gsl_complex_rect (InvPmRe, InvPmIm));
                    }
                }
                
                // Matrix inversion (1 band)
                
                gsl_permutation *permPp = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Pp = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvPp, permPp, &perm);
                gsl_linalg_complex_LU_invert (InvPp, permPp, Pp);
                
                gsl_permutation *permPm = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Pm = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvPm, permPm, &perm);
                gsl_linalg_complex_LU_invert (InvPm, permPm, Pm);
                
                // Free the previous allocation
                
                gsl_permutation_free (permPp);
                gsl_matrix_complex_free (InvPp);
                
                gsl_permutation_free (permPm);
                gsl_matrix_complex_free (InvPm);
                
                
                /************************************************/
                /*** 2-band retarded lattice Green's function ***/
                /************************************************/
                
                gsl_matrix_complex *InvQa = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvQbp = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvQbm = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        Pmab = gsl_matrix_complex_get (Pm, p, q);
                        Ppab = gsl_matrix_complex_get (Pp, p, q);
                        
                        if (p == 0) {
                            
                            Ppamb = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppamb = gsl_matrix_complex_get (Pp, p-1, q);
                        }
                        
                        if (p == Ncut-1) {
                            
                            Ppapb = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppapb = gsl_matrix_complex_get (Pp, p+1, q);
                        }
                        
                        if (q == 0) {
                            
                            Ppabm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppabm = gsl_matrix_complex_get (Pp, p, q-1);
                        }
                        
                        if (q == Ncut-1) {
                            
                            Ppabp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppabp = gsl_matrix_complex_get (Pp, p, q+1);
                        }
                        
                        if (p == 0 || q == 0) {
                            
                            Ppambm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppambm = gsl_matrix_complex_get (Pp, p-1, q-1);
                        }
                        
                        if (p == Ncut-1 || q == Ncut-1) {
                            
                            Ppapbp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppapbp = gsl_matrix_complex_get (Pp, p+1, q+1);
                        }
                        
                        if (p == 0 || q == Ncut-1) {
                            
                            Ppambp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppambp = gsl_matrix_complex_get (Pp, p-1, q+1);
                        }
                        
                        if (p == Ncut-1 || q == 0) {
                            
                            Ppapbm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Ppapbm = gsl_matrix_complex_get (Pp, p+1, q-1);
                        }
                        
                        MixQaRe = pow(A2*sin(Kperp),2) * GSL_REAL(Pmab);
                        MixQaIm = pow(A2*sin(Kperp),2) * GSL_IMAG(Pmab);
                        
                        MixQbpRe = pow(0.5*A1,2) * ( pow(2.0*sin(Kplane),2) * GSL_REAL(Ppab)
                                 - 2.0*sin(Kplane) * ( GSL_REAL(Ppamb) - GSL_REAL(Ppapb) + GSL_REAL(Ppabm) - GSL_REAL(Ppabp) )
                                 + GSL_REAL(Ppambm) + GSL_REAL(Ppapbp) - GSL_REAL(Ppambp) - GSL_REAL(Ppapbm) );
                        MixQbpIm = pow(0.5*A1,2) * ( pow(2.0*sin(Kplane),2) * GSL_IMAG(Ppab)
                                 - 2.0*sin(Kplane) * ( GSL_IMAG(Ppamb) - GSL_IMAG(Ppapb) + GSL_IMAG(Ppabm) - GSL_IMAG(Ppabp) )
                                 + GSL_IMAG(Ppambm) + GSL_IMAG(Ppapbp) - GSL_IMAG(Ppambp) - GSL_IMAG(Ppapbm) );
                        
                        MixQbmRe = pow(0.5*A1,2) * ( pow(2.0*sin(Kplane),2) * GSL_REAL(Ppab)
                                 - 2.0*sin(Kplane) * ( GSL_REAL(Ppapb) - GSL_REAL(Ppamb) + GSL_REAL(Ppabp) - GSL_REAL(Ppabm) )
                                 + GSL_REAL(Ppapbp) + GSL_REAL(Ppambm) - GSL_REAL(Ppapbm) - GSL_REAL(Ppambp) );
                        MixQbmIm = pow(0.5*A1,2) * ( pow(2.0*sin(Kplane),2) * GSL_IMAG(Ppab)
                                 - 2.0*sin(Kplane) * ( GSL_IMAG(Ppapb) - GSL_IMAG(Ppamb) + GSL_IMAG(Ppabp) - GSL_IMAG(Ppabm) )
                                 + GSL_IMAG(Ppapbp) + GSL_IMAG(Ppambm) - GSL_IMAG(Ppapbm) - GSL_IMAG(Ppambp) );
                        
                        if (p == q) {
                            
                            InvQaRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                    - (C + M) - 2.0*(D2 + B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 + B1) * (2.0 - cos(Kplane)) - MixQaRe;
                            InvQaIm = 0.5*Gamma - MixQaIm;
                            
                            InvQbpRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                     - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 - B1) * (2.0 - cos(Kplane)) - MixQbpRe;
                            InvQbpIm = 0.5*Gamma - MixQbpIm;
                            
                            InvQbmRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                     - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 - B1) * (2.0 - cos(Kplane)) - MixQbmRe;
                            InvQbmIm = 0.5*Gamma - MixQbmIm;
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            InvQaRe = 0.25 + B1 - MixQaRe;
                            InvQaIm = - MixQaIm;
                            
                            InvQbpRe = 0.25 - B1 - MixQbpRe;
                            InvQbpIm = - MixQbpIm;
                            
                            InvQbmRe = 0.25 - B1 - MixQbmRe;
                            InvQbmIm = - MixQbmIm;
                            
                        } else {
                            
                            InvQaRe = - MixQaRe;
                            InvQaIm = - MixQaIm;
                            
                            InvQbpRe = - MixQbpRe;
                            InvQbpIm = - MixQbpIm;
                            
                            InvQbmRe = - MixQbmRe;
                            InvQbmIm = - MixQbmIm;
                        }
                        
                        gsl_matrix_complex_set (InvQa, p, q, gsl_complex_rect (InvQaRe, InvQaIm));
                        gsl_matrix_complex_set (InvQbp, p, q, gsl_complex_rect (InvQbpRe, InvQbpIm));
                        gsl_matrix_complex_set (InvQbm, p, q, gsl_complex_rect (InvQbmRe, InvQbmIm));
                    }
                }
                
                // Matrix inversion (2 bands)
                
                gsl_permutation *permQa = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Qa = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvQa, permQa, &perm);
                gsl_linalg_complex_LU_invert (InvQa, permQa, Qa);
                
                gsl_permutation *permQbp = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Qbp = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvQbp, permQbp, &perm);
                gsl_linalg_complex_LU_invert (InvQbp, permQbp, Qbp);
                
                gsl_permutation *permQbm = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Qbm = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvQbm, permQbm, &perm);
                gsl_linalg_complex_LU_invert (InvQbm, permQbm, Qbm);
                
                // Free the previous allocation
                
                gsl_permutation_free (permQa);
                gsl_matrix_complex_free (InvQa);
                
                gsl_permutation_free (permQbp);
                gsl_matrix_complex_free (InvQbp);
                
                gsl_permutation_free (permQbm);
                gsl_matrix_complex_free (InvQbm);
                
                
                /************************************************/
                /*** 3-band retarded lattice Green's function ***/
                /************************************************/
                
                gsl_matrix_complex *InvRap = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvRam = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvRbp = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvRbm = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        Ppab = gsl_matrix_complex_get (Pp, p, q);
                        Pmab = gsl_matrix_complex_get (Pm, p, q);
                        
                        if (p == 0) {
                            
                            Pmamb = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmamb = gsl_matrix_complex_get (Pm, p-1, q);
                        }
                        
                        if (p == Ncut-1) {
                            
                            Pmapb = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmapb = gsl_matrix_complex_get (Pm, p+1, q);
                        }
                        
                        if (q == 0) {
                            
                            Pmabm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmabm = gsl_matrix_complex_get (Pm, p, q-1);
                        }
                        
                        if (q == Ncut-1) {
                            
                            Pmabp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmabp = gsl_matrix_complex_get (Pm, p, q+1);
                        }
                        
                        if (p == 0 || q == 0) {
                            
                            Pmambm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmambm = gsl_matrix_complex_get (Pm, p-1, q-1);
                        }
                        
                        if (p == Ncut-1 || q == Ncut-1) {
                            
                            Pmapbp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmapbp = gsl_matrix_complex_get (Pm, p+1, q+1);
                        }
                        
                        if (p == 0 || q == Ncut-1) {
                            
                            Pmambp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmambp = gsl_matrix_complex_get (Pm, p-1, q+1);
                        }
                        
                        if (p == Ncut-1 || q == 0) {
                            
                            Pmapbm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Pmapbm = gsl_matrix_complex_get (Pm, p+1, q-1);
                        }
                        
                        MixRapRe = pow(0.5*A1,2) * ( pow(2.0*sin(Kplane),2) * GSL_REAL(Pmab)
                                 - 2.0*sin(Kplane) * ( GSL_REAL(Pmamb) - GSL_REAL(Pmapb) + GSL_REAL(Pmabm) - GSL_REAL(Pmabp) )
                                 + GSL_REAL(Pmambm) + GSL_REAL(Pmapbp) - GSL_REAL(Pmambp) - GSL_REAL(Pmapbm) );
                        MixRapIm = pow(0.5*A1,2) * ( pow(2.0*sin(Kplane),2) * GSL_IMAG(Pmab)
                                 - 2.0*sin(Kplane) * ( GSL_IMAG(Pmamb) - GSL_IMAG(Pmapb) + GSL_IMAG(Pmabm) - GSL_IMAG(Pmabp) )
                                 + GSL_IMAG(Pmambm) + GSL_IMAG(Pmapbp) - GSL_IMAG(Pmambp) - GSL_IMAG(Pmapbm) );
                        
                        MixRamRe = pow(0.5*A1,2) * ( pow(2.0*sin(Kplane),2) * GSL_REAL(Pmab)
                                 - 2.0*sin(Kplane) * ( GSL_REAL(Pmapb) - GSL_REAL(Pmamb) + GSL_REAL(Pmabp) - GSL_REAL(Pmabm) )
                                 + GSL_REAL(Pmapbp) + GSL_REAL(Pmambm) - GSL_REAL(Pmapbm) - GSL_REAL(Pmambp) );
                        MixRamIm = pow(0.5*A1,2) * ( pow(2.0*sin(Kplane),2) * GSL_IMAG(Pmab)
                                 - 2.0*sin(Kplane) * ( GSL_IMAG(Pmapb) - GSL_IMAG(Pmamb) + GSL_IMAG(Pmabp) - GSL_IMAG(Pmabm) )
                                 + GSL_IMAG(Pmapbp) + GSL_IMAG(Pmambm) - GSL_IMAG(Pmapbm) - GSL_IMAG(Pmambp) );
                        
                        MixRbpRe = MixRbmRe = pow(A2*sin(Kperp),2) * GSL_REAL(Ppab);
                        MixRbpIm = MixRbmIm = pow(A2*sin(Kperp),2) * GSL_IMAG(Ppab);
                        
                        for (r = 0; r <= Ncut-1; r++)
                        {
                            for (s = 0; s <= Ncut-1; s++)
                            {
                                PmLab = gsl_matrix_complex_get (Pm, p, r);
                                PmRab = gsl_matrix_complex_get (Pm, s, q);
                                
                                if (p == 0) {
                                    
                                    PmLamb = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PmLamb = gsl_matrix_complex_get (Pm, p-1, r);
                                }
                                
                                if (p == Ncut-1) {
                                    
                                    PmLapb = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PmLapb = gsl_matrix_complex_get (Pm, p+1, r);
                                }
                                
                                if (q == 0) {
                                    
                                    PmRabm = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PmRabm = gsl_matrix_complex_get (Pm, s, q-1);
                                }
                                
                                if (q == Ncut-1) {
                                    
                                    PmRabp = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PmRabp = gsl_matrix_complex_get (Pm, s, q+1);
                                }
                                
                                PpLab = gsl_matrix_complex_get (Pp, p, r);
                                PpRab = gsl_matrix_complex_get (Pp, s, q);
                                
                                if (r == 0) {
                                    
                                    PpLabm = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PpLabm = gsl_matrix_complex_get (Pp, p, r-1);
                                }
                                
                                if (r == Ncut-1) {
                                    
                                    PpLabp = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PpLabp = gsl_matrix_complex_get (Pp, p, r+1);
                                }
                                
                                if (s == 0) {
                                    
                                    PpRamb = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PpRamb = gsl_matrix_complex_get (Pp, s-1, q);
                                }
                                
                                if (s == Ncut-1) {
                                    
                                    PpRapb = gsl_complex_rect (0, 0);
                                    
                                } else {
                                    
                                    PpRapb = gsl_matrix_complex_get (Pp, s+1, q);
                                }
                                
                                Qaab = gsl_matrix_complex_get (Qa, r, s);
                                Qbpab = gsl_matrix_complex_get (Qbp, r, s);
                                Qbmab = gsl_matrix_complex_get (Qbm, r, s);
                                
                                MixRapRe += pow(0.5*A1*A2*sin(Kperp),2)
                                         * ( ( ( 2.0*sin(Kplane) * GSL_REAL(PmLab) - GSL_REAL(PmLamb) + GSL_REAL(PmLapb) ) * GSL_REAL(Qaab)
                                         - ( 2.0*sin(Kplane) * GSL_IMAG(PmLab) - GSL_IMAG(PmLamb) + GSL_IMAG(PmLapb) ) * GSL_IMAG(Qaab) )
                                         * ( 2.0*sin(Kplane) * GSL_REAL(PmRab) - GSL_REAL(PmRabm) + GSL_REAL(PmRabp) )
                                         - ( ( 2.0*sin(Kplane) * GSL_REAL(PmLab) - GSL_REAL(PmLamb) + GSL_REAL(PmLapb) ) * GSL_IMAG(Qaab)
                                         + ( 2.0*sin(Kplane) * GSL_IMAG(PmLab) - GSL_IMAG(PmLamb) + GSL_IMAG(PmLapb) ) * GSL_REAL(Qaab) )
                                         * ( 2.0*sin(Kplane) * GSL_IMAG(PmRab) - GSL_IMAG(PmRabm) + GSL_IMAG(PmRabp) ) );
                                
                                MixRapIm += pow(0.5*A1*A2*sin(Kperp),2)
                                         * ( ( ( 2.0*sin(Kplane) * GSL_REAL(PmLab) - GSL_REAL(PmLamb) + GSL_REAL(PmLapb) ) * GSL_REAL(Qaab)
                                         - ( 2.0*sin(Kplane) * GSL_IMAG(PmLab) - GSL_IMAG(PmLamb) + GSL_IMAG(PmLapb) ) * GSL_IMAG(Qaab) )
                                         * ( 2.0*sin(Kplane) * GSL_IMAG(PmRab) - GSL_IMAG(PmRabm) + GSL_IMAG(PmRabp) )
                                         + ( ( 2.0*sin(Kplane) * GSL_REAL(PmLab) - GSL_REAL(PmLamb) + GSL_REAL(PmLapb) ) * GSL_IMAG(Qaab)
                                         + ( 2.0*sin(Kplane) * GSL_IMAG(PmLab) - GSL_IMAG(PmLamb) + GSL_IMAG(PmLapb) ) * GSL_REAL(Qaab) )
                                         * ( 2.0*sin(Kplane) * GSL_REAL(PmRab) - GSL_REAL(PmRabm) + GSL_REAL(PmRabp) ) );
                                
                                MixRamRe += pow(0.5*A1*A2*sin(Kperp),2)
                                         * ( ( ( 2.0*sin(Kplane) * GSL_REAL(PmLab) - GSL_REAL(PmLapb) + GSL_REAL(PmLamb) ) * GSL_REAL(Qaab)
                                         - ( 2.0*sin(Kplane) * GSL_IMAG(PmLab) - GSL_IMAG(PmLapb) + GSL_IMAG(PmLamb) ) * GSL_IMAG(Qaab) )
                                         * ( 2.0*sin(Kplane) * GSL_REAL(PmRab) - GSL_REAL(PmRabp) + GSL_REAL(PmRabm) )
                                         - ( ( 2.0*sin(Kplane) * GSL_REAL(PmLab) - GSL_REAL(PmLapb) + GSL_REAL(PmLamb) ) * GSL_IMAG(Qaab)
                                         + ( 2.0*sin(Kplane) * GSL_IMAG(PmLab) - GSL_IMAG(PmLapb) + GSL_IMAG(PmLamb) ) * GSL_REAL(Qaab) )
                                         * ( 2.0*sin(Kplane) * GSL_IMAG(PmRab) - GSL_IMAG(PmRabp) + GSL_IMAG(PmRabm) ) );
                                
                                MixRamIm += pow(0.5*A1*A2*sin(Kperp),2)
                                         * ( ( ( 2.0*sin(Kplane) * GSL_REAL(PmLab) - GSL_REAL(PmLapb) + GSL_REAL(PmLamb) ) * GSL_REAL(Qaab)
                                         - ( 2.0*sin(Kplane) * GSL_IMAG(PmLab) - GSL_IMAG(PmLapb) + GSL_IMAG(PmLamb) ) * GSL_IMAG(Qaab) )
                                         * ( 2.0*sin(Kplane) * GSL_IMAG(PmRab) - GSL_IMAG(PmRabp) + GSL_IMAG(PmRabm) )
                                         + ( ( 2.0*sin(Kplane) * GSL_REAL(PmLab) - GSL_REAL(PmLapb) + GSL_REAL(PmLamb) ) * GSL_IMAG(Qaab)
                                         + ( 2.0*sin(Kplane) * GSL_IMAG(PmLab) - GSL_IMAG(PmLapb) + GSL_IMAG(PmLamb) ) * GSL_REAL(Qaab) )
                                         * ( 2.0*sin(Kplane) * GSL_REAL(PmRab) - GSL_REAL(PmRabp) + GSL_REAL(PmRabm) ) );
                                
                                MixRbpRe += pow(0.5*A1*A2*sin(Kperp),2)
                                         * ( ( ( 2.0*sin(Kplane) * GSL_REAL(PpLab) - GSL_REAL(PpLabm) + GSL_REAL(PpLabp) ) * GSL_REAL(Qbpab)
                                         - ( 2.0*sin(Kplane) * GSL_IMAG(PpLab) - GSL_IMAG(PpLabm) + GSL_IMAG(PpLabp) ) * GSL_IMAG(Qbpab) )
                                         * ( 2.0*sin(Kplane) * GSL_REAL(PpRab) - GSL_REAL(PpRamb) + GSL_REAL(PpRapb) )
                                         - ( ( 2.0*sin(Kplane) * GSL_REAL(PpLab) - GSL_REAL(PpLabm) + GSL_REAL(PpLabp) ) * GSL_IMAG(Qbpab)
                                         + ( 2.0*sin(Kplane) * GSL_IMAG(PpLab) - GSL_IMAG(PpLabm) + GSL_IMAG(PpLabp) ) * GSL_REAL(Qbpab) )
                                         * ( 2.0*sin(Kplane) * GSL_IMAG(PpRab) - GSL_IMAG(PpRamb) + GSL_IMAG(PpRapb) ) );
                                
                                MixRbpIm += pow(0.5*A1*A2*sin(Kperp),2)
                                         * ( ( ( 2.0*sin(Kplane) * GSL_REAL(PpLab) - GSL_REAL(PpLabm) + GSL_REAL(PpLabp) ) * GSL_REAL(Qbpab)
                                         - ( 2.0*sin(Kplane) * GSL_IMAG(PpLab) - GSL_IMAG(PpLabm) + GSL_IMAG(PpLabp) ) * GSL_IMAG(Qbpab) )
                                         * ( 2.0*sin(Kplane) * GSL_IMAG(PpRab) - GSL_IMAG(PpRamb) + GSL_IMAG(PpRapb) )
                                         + ( ( 2.0*sin(Kplane) * GSL_REAL(PpLab) - GSL_REAL(PpLabm) + GSL_REAL(PpLabp) ) * GSL_IMAG(Qbpab)
                                         + ( 2.0*sin(Kplane) * GSL_IMAG(PpLab) - GSL_IMAG(PpLabm) + GSL_IMAG(PpLabp) ) * GSL_REAL(Qbpab) )
                                         * ( 2.0*sin(Kplane) * GSL_REAL(PpRab) - GSL_REAL(PpRamb) + GSL_REAL(PpRapb) ) );
                                
                                MixRbmRe += pow(0.5*A1*A2*sin(Kperp),2)
                                         * ( ( ( 2.0*sin(Kplane) * GSL_REAL(PpLab) - GSL_REAL(PpLabp) + GSL_REAL(PpLabm) ) * GSL_REAL(Qbmab)
                                         - ( 2.0*sin(Kplane) * GSL_IMAG(PpLab) - GSL_IMAG(PpLabp) + GSL_IMAG(PpLabm) ) * GSL_IMAG(Qbmab) )
                                         * ( 2.0*sin(Kplane) * GSL_REAL(PpRab) - GSL_REAL(PpRapb) + GSL_REAL(PpRamb) )
                                         - ( ( 2.0*sin(Kplane) * GSL_REAL(PpLab) - GSL_REAL(PpLabp) + GSL_REAL(PpLabm) ) * GSL_IMAG(Qbmab)
                                         + ( 2.0*sin(Kplane) * GSL_IMAG(PpLab) - GSL_IMAG(PpLabp) + GSL_IMAG(PpLabm) ) * GSL_REAL(Qbmab) )
                                         * ( 2.0*sin(Kplane) * GSL_IMAG(PpRab) - GSL_IMAG(PpRapb) + GSL_IMAG(PpRamb) ) );
                                
                                MixRbmIm += pow(0.5*A1*A2*sin(Kperp),2)
                                         * ( ( ( 2.0*sin(Kplane) * GSL_REAL(PpLab) - GSL_REAL(PpLabp) + GSL_REAL(PpLabm) ) * GSL_REAL(Qbmab)
                                         - ( 2.0*sin(Kplane) * GSL_IMAG(PpLab) - GSL_IMAG(PpLabp) + GSL_IMAG(PpLabm) ) * GSL_IMAG(Qbmab) )
                                         * ( 2.0*sin(Kplane) * GSL_IMAG(PpRab) - GSL_IMAG(PpRapb) + GSL_IMAG(PpRamb) )
                                         + ( ( 2.0*sin(Kplane) * GSL_REAL(PpLab) - GSL_REAL(PpLabp) + GSL_REAL(PpLabm) ) * GSL_IMAG(Qbmab)
                                         + ( 2.0*sin(Kplane) * GSL_IMAG(PpLab) - GSL_IMAG(PpLabp) + GSL_IMAG(PpLabm) ) * GSL_REAL(Qbmab) )
                                         * ( 2.0*sin(Kplane) * GSL_REAL(PpRab) - GSL_REAL(PpRapb) + GSL_REAL(PpRamb) ) );
                            }
                        }
                        
                        if (p == q) {
                            
                            InvRapRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                     - (C + M) - 2.0*(D2 + B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 + B1) * (2.0 - cos(Kplane)) - MixRapRe;
                            InvRapIm = 0.5*Gamma - MixRapIm;
                            
                            InvRamRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                     - (C + M) - 2.0*(D2 + B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 + B1) * (2.0 - cos(Kplane)) - MixRamRe;
                            InvRamIm = 0.5*Gamma - MixRamIm;
                            
                            InvRbpRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                     - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 - B1) * (2.0 - cos(Kplane)) - MixRbpRe;
                            InvRbpIm = 0.5*Gamma - MixRbpIm;
                            
                            InvRbmRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                     - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 - B1) * (2.0 - cos(Kplane)) - MixRbmRe;
                            InvRbmIm = 0.5*Gamma - MixRbmIm;
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            InvRapRe = 0.25 + B1 - MixRapRe;
                            InvRapIm = - MixRapIm;
                            
                            InvRamRe = 0.25 + B1 - MixRamRe;
                            InvRamIm = - MixRamIm;
                            
                            InvRbpRe = 0.25 - B1 - MixRbpRe;
                            InvRbpIm = - MixRbpIm;
                            
                            InvRbmRe = 0.25 - B1 - MixRbmRe;
                            InvRbmIm = - MixRbmIm;
                            
                        } else {
                            
                            InvRapRe = - MixRapRe;
                            InvRapIm = - MixRapIm;
                            
                            InvRamRe = - MixRamRe;
                            InvRamIm = - MixRamIm;
                            
                            InvRbpRe = - MixRbpRe;
                            InvRbpIm = - MixRbpIm;
                            
                            InvRbmRe = - MixRbmRe;
                            InvRbmIm = - MixRbmIm;
                        }
                        
                        gsl_matrix_complex_set (InvRap, p, q, gsl_complex_rect (InvRapRe, InvRapIm));
                        gsl_matrix_complex_set (InvRam, p, q, gsl_complex_rect (InvRamRe, InvRamIm));
                        gsl_matrix_complex_set (InvRbp, p, q, gsl_complex_rect (InvRbpRe, InvRbpIm));
                        gsl_matrix_complex_set (InvRbm, p, q, gsl_complex_rect (InvRbmRe, InvRbmIm));
                    }
                }
                
                // Matrix inversion (3 bands)
                
                gsl_permutation *permRap = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Rap = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvRap, permRap, &perm);
                gsl_linalg_complex_LU_invert (InvRap, permRap, Rap);
                
                gsl_permutation *permRam = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Ram = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvRam, permRam, &perm);
                gsl_linalg_complex_LU_invert (InvRam, permRam, Ram);
                
                gsl_permutation *permRbp = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Rbp = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvRbp, permRbp, &perm);
                gsl_linalg_complex_LU_invert (InvRbp, permRbp, Rbp);
                
                gsl_permutation *permRbm = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Rbm = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvRbm, permRbm, &perm);
                gsl_linalg_complex_LU_invert (InvRbm, permRbm, Rbm);
                
                // Free the previous allocation
                
                gsl_permutation_free (permRap);
                gsl_matrix_complex_free (InvRap);
                
                gsl_permutation_free (permRam);
                gsl_matrix_complex_free (InvRam);
                
                gsl_permutation_free (permRbp);
                gsl_matrix_complex_free (InvRbp);
                
                gsl_permutation_free (permRbm);
                gsl_matrix_complex_free (InvRbm);
                
                
                /************************************************/
                /*** 4-band retarded lattice Green's function ***/
                /************************************************/
                
                gsl_matrix_complex *MaL = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *MaR = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *MbL = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *MbR = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *McL = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *McR = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *MdL = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *MdR = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                gsl_matrix_complex *InvGa = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvGb = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvGc = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_matrix_complex *InvGd = gsl_matrix_complex_alloc (Ncut, Ncut);
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        if (p == q) {
                            
                            MaLRe = MaRRe = A2*sin(Kperp);
                            MaLIm = MaRIm = 0;
                            
                            McLRe = McRRe = A2*sin(Kperp);
                            McLIm = McRIm = 0;
                            
                        } else {
                            
                            MaLRe = MaRRe = 0;
                            MaLIm = MaRIm = 0;
                            
                            McLRe = McRRe = 0;
                            McLIm = McRIm = 0;
                        }
                        
                        if (p == q) {
                            
                            MbLRe = MbRRe = A1*sin(Kplane);
                            MbLIm = MbRIm = 0;
                            
                            MdLRe = MdRRe = A1*sin(Kplane);
                            MdLIm = MdRIm = 0;
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            MbLRe = -(p-q)*0.5*A1;
                            MbRRe = (p-q)*0.5*A1;
                            MbLIm = MbRIm = 0;
                            
                            MdLRe = (p-q)*0.5*A1;
                            MdRRe = -(p-q)*0.5*A1;
                            MdLIm = MdRIm = 0;
                            
                        } else {
                            
                            MbLRe = MbRRe = 0;
                            MbLIm = MbRIm = 0;
                            
                            MdLRe = MdRRe = 0;
                            MdLIm = MdRIm = 0;
                        }
                        
                        for (r = 0; r <= Ncut-1; r++)
                        {
                            QaLab = gsl_matrix_complex_get (Qa, p, r);
                            QaRab = gsl_matrix_complex_get (Qa, r, q);
                            
                            QbpLab = gsl_matrix_complex_get (Qbp, p, r);
                            QbpRab = gsl_matrix_complex_get (Qbp, r, q);
                            
                            QbmLab = gsl_matrix_complex_get (Qbm, p, r);
                            QbmRab = gsl_matrix_complex_get (Qbm, r, q);
                            
                            PpLab = gsl_matrix_complex_get (Pp, p, r);
                            PpRab = gsl_matrix_complex_get (Pp, r, q);
                            
                            PmLab = gsl_matrix_complex_get (Pm, p, r);
                            PmRab = gsl_matrix_complex_get (Pm, r, q);
                            
                            if (p == 0) {
                                
                                Qaamb = gsl_complex_rect (0, 0);
                                Pmamb = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                Qaamb = gsl_matrix_complex_get (Qa, p-1, r);
                                Pmamb = gsl_matrix_complex_get (Pm, p-1, r);
                            }
                            
                            if (p == Ncut-1) {
                                
                                Qaapb = gsl_complex_rect (0, 0);
                                Pmapb = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                Qaapb = gsl_matrix_complex_get (Qa, p+1, r);
                                Pmapb = gsl_matrix_complex_get (Pm, p+1, r);
                            }
                            
                            if (q == 0) {
                                
                                Qaabm = gsl_complex_rect (0, 0);
                                Pmabm = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                Qaabm = gsl_matrix_complex_get (Qa, r, q-1);
                                Pmabm = gsl_matrix_complex_get (Pm, r, q-1);
                            }
                            
                            if (q == Ncut-1) {
                                
                                Qaabp = gsl_complex_rect (0, 0);
                                Pmabp = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                Qaabp = gsl_matrix_complex_get (Qa, r, q+1);
                                Pmabp = gsl_matrix_complex_get (Pm, r, q+1);
                            }
                            
                            if (r == 0) {
                                
                                Qbpabm = gsl_complex_rect (0, 0);
                                Qbmabm = gsl_complex_rect (0, 0);
                                Ppabm = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                Qbpabm = gsl_matrix_complex_get (Qbp, p, r-1);
                                Qbmabm = gsl_matrix_complex_get (Qbm, p, r-1);
                                Ppabm = gsl_matrix_complex_get (Pp, p, r-1);
                            }
                            
                            if (r == Ncut-1) {
                                
                                Qbpabp = gsl_complex_rect (0, 0);
                                Qbmabp = gsl_complex_rect (0, 0);
                                Ppabp = gsl_complex_rect (0, 0);
                                
                            } else {
                                
                                Qbpabp = gsl_matrix_complex_get (Qbp, p, r+1);
                                Qbmabp = gsl_matrix_complex_get (Qbm, p, r+1);
                                Ppabp = gsl_matrix_complex_get (Pp, p, r+1);
                            }
                            
                            MaLRe += - pow(0.5*A1,2) * A2*sin(Kperp)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(QaLab) - GSL_REAL(Qaamb) + GSL_REAL(Qaapb) )
                                  * ( 2.0*sin(Kplane) * GSL_REAL(PmRab) - GSL_REAL(Pmabm) + GSL_REAL(Pmabp) )
                                  - ( 2.0*sin(Kplane) * GSL_IMAG(QaLab) - GSL_IMAG(Qaamb) + GSL_IMAG(Qaapb) )
                                  * ( 2.0*sin(Kplane) * GSL_IMAG(PmRab) - GSL_IMAG(Pmabm) + GSL_IMAG(Pmabp) ) );
                            MaLIm += - pow(0.5*A1,2) * A2*sin(Kperp)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(QaLab) - GSL_REAL(Qaamb) + GSL_REAL(Qaapb) )
                                  * ( 2.0*sin(Kplane) * GSL_IMAG(PmRab) - GSL_IMAG(Pmabm) + GSL_IMAG(Pmabp) )
                                  + ( 2.0*sin(Kplane) * GSL_IMAG(QaLab) - GSL_IMAG(Qaamb) + GSL_IMAG(Qaapb) )
                                  * ( 2.0*sin(Kplane) * GSL_REAL(PmRab) - GSL_REAL(Pmabm) + GSL_REAL(Pmabp) ) );
                            
                            MaRRe += - pow(0.5*A1,2) * A2*sin(Kperp)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(PmLab) - GSL_REAL(Pmamb) + GSL_REAL(Pmapb) )
                                  * ( 2.0*sin(Kplane) * GSL_REAL(QaRab) - GSL_REAL(Qaabm) + GSL_REAL(Qaabp) )
                                  - ( 2.0*sin(Kplane) * GSL_IMAG(PmLab) - GSL_IMAG(Pmamb) + GSL_IMAG(Pmapb) )
                                  * ( 2.0*sin(Kplane) * GSL_IMAG(QaRab) - GSL_IMAG(Qaabm) + GSL_IMAG(Qaabp) ) );
                            MaRIm += - pow(0.5*A1,2) * A2*sin(Kperp)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(PmLab) - GSL_REAL(Pmamb) + GSL_REAL(Pmapb) )
                                  * ( 2.0*sin(Kplane) * GSL_IMAG(QaRab) - GSL_IMAG(Qaabm) + GSL_IMAG(Qaabp) )
                                  + ( 2.0*sin(Kplane) * GSL_IMAG(PmLab) - GSL_IMAG(Pmamb) + GSL_IMAG(Pmapb) )
                                  * ( 2.0*sin(Kplane) * GSL_REAL(QaRab) - GSL_REAL(Qaabm) + GSL_REAL(Qaabp) ) );
                            
                            MbLRe += - 0.5*A1 * pow(A2*sin(Kperp),2)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(QbpLab) - GSL_REAL(Qbpabp) + GSL_REAL(Qbpabm) ) * GSL_REAL(PpRab)
                                  - ( 2.0*sin(Kplane) * GSL_IMAG(QbpLab) - GSL_IMAG(Qbpabp) + GSL_IMAG(Qbpabm) ) * GSL_IMAG(PpRab) );
                            MbLIm += - 0.5*A1 * pow(A2*sin(Kperp),2)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(QbpLab) - GSL_REAL(Qbpabp) + GSL_REAL(Qbpabm) ) * GSL_IMAG(PpRab)
                                  + ( 2.0*sin(Kplane) * GSL_IMAG(QbpLab) - GSL_IMAG(Qbpabp) + GSL_IMAG(Qbpabm) ) * GSL_REAL(PpRab) );
                            
                            MbRRe += - 0.5*A1 * pow(A2*sin(Kperp),2)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(PpLab) - GSL_REAL(Ppabm) + GSL_REAL(Ppabp) ) * GSL_REAL(QbpRab)
                                  - ( 2.0*sin(Kplane) * GSL_IMAG(PpLab) - GSL_IMAG(Ppabm) + GSL_IMAG(Ppabp) ) * GSL_IMAG(QbpRab) );
                            MbRIm += - 0.5*A1 * pow(A2*sin(Kperp),2)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(PpLab) - GSL_REAL(Ppabm) + GSL_REAL(Ppabp) ) * GSL_IMAG(QbpRab)
                                  + ( 2.0*sin(Kplane) * GSL_IMAG(PpLab) - GSL_IMAG(Ppabm) + GSL_IMAG(Ppabp) ) * GSL_REAL(QbpRab) );
                            
                            McLRe += - pow(0.5*A1,2) * A2*sin(Kperp)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(QaLab) - GSL_REAL(Qaapb) + GSL_REAL(Qaamb) )
                                  * ( 2.0*sin(Kplane) * GSL_REAL(PmRab) - GSL_REAL(Pmabp) + GSL_REAL(Pmabm) )
                                  - ( 2.0*sin(Kplane) * GSL_IMAG(QaLab) - GSL_IMAG(Qaapb) + GSL_IMAG(Qaamb) )
                                  * ( 2.0*sin(Kplane) * GSL_IMAG(PmRab) - GSL_IMAG(Pmabp) + GSL_IMAG(Pmabm) ) );
                            McLIm += - pow(0.5*A1,2) * A2*sin(Kperp)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(QaLab) - GSL_REAL(Qaapb) + GSL_REAL(Qaamb) )
                                  * ( 2.0*sin(Kplane) * GSL_IMAG(PmRab) - GSL_IMAG(Pmabp) + GSL_IMAG(Pmabm) )
                                  + ( 2.0*sin(Kplane) * GSL_IMAG(QaLab) - GSL_IMAG(Qaapb) + GSL_IMAG(Qaamb) )
                                 * ( 2.0*sin(Kplane) * GSL_REAL(PmRab) - GSL_REAL(Pmabp) + GSL_REAL(Pmabm) ) );
                            
                            McRRe += - pow(0.5*A1,2) * A2*sin(Kperp)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(PmLab) - GSL_REAL(Pmapb) + GSL_REAL(Pmamb) )
                                  * ( 2.0*sin(Kplane) * GSL_REAL(QaRab) - GSL_REAL(Qaabp) + GSL_REAL(Qaabm) )
                                  - ( 2.0*sin(Kplane) * GSL_IMAG(PmLab) - GSL_IMAG(Pmapb) + GSL_IMAG(Pmamb) )
                                  * ( 2.0*sin(Kplane) * GSL_IMAG(QaRab) - GSL_IMAG(Qaabp) + GSL_IMAG(Qaabm) ) );
                            McRIm += - pow(0.5*A1,2) * A2*sin(Kperp)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(PmLab) - GSL_REAL(Pmapb) + GSL_REAL(Pmamb) )
                                  * ( 2.0*sin(Kplane) * GSL_IMAG(QaRab) - GSL_IMAG(Qaabp) + GSL_IMAG(Qaabm) )
                                  + ( 2.0*sin(Kplane) * GSL_IMAG(PmLab) - GSL_IMAG(Pmapb) + GSL_IMAG(Pmamb) )
                                  * ( 2.0*sin(Kplane) * GSL_REAL(QaRab) - GSL_REAL(Qaabp) + GSL_REAL(Qaabm) ) );
                            
                            MdLRe += - 0.5*A1 * pow(A2*sin(Kperp),2)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(QbmLab) - GSL_REAL(Qbmabm) + GSL_REAL(Qbmabp) ) * GSL_REAL(PpRab)
                                  - ( 2.0*sin(Kplane) * GSL_IMAG(QbmLab) - GSL_IMAG(Qbmabm) + GSL_IMAG(Qbmabp) ) * GSL_IMAG(PpRab) );
                            MdLIm += - 0.5*A1 * pow(A2*sin(Kplane),2)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(QbmLab) - GSL_REAL(Qbmabm) + GSL_REAL(Qbmabp) ) * GSL_IMAG(PpRab)
                                  + ( 2.0*sin(Kplane) * GSL_IMAG(QbmLab) - GSL_IMAG(Qbmabm) + GSL_IMAG(Qbmabp) ) * GSL_REAL(PpRab) );
                            
                            MdRRe += - 0.5*A1 * pow(A2*sin(Kperp),2)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(PpLab) - GSL_REAL(Ppabp) + GSL_REAL(Ppabm) ) * GSL_REAL(QbmRab)
                                  - ( 2.0*sin(Kplane) * GSL_IMAG(PpLab) - GSL_IMAG(Ppabp) + GSL_IMAG(Ppabm) ) * GSL_IMAG(QbmRab) );
                            MdRIm += - 0.5*A1 * pow(A2*sin(Kplane),2)
                                  * ( ( 2.0*sin(Kplane) * GSL_REAL(PpLab) - GSL_REAL(Ppabp) + GSL_REAL(Ppabm) ) * GSL_IMAG(QbmRab)
                                  + ( 2.0*sin(Kplane) * GSL_IMAG(PpLab) - GSL_IMAG(Ppabp) + GSL_IMAG(Ppabm) ) * GSL_REAL(QbmRab) );
                        }
                        
                        gsl_matrix_complex_set (MaL, p, q, gsl_complex_rect (MaLRe, MaLIm));
                        gsl_matrix_complex_set (MaR, p, q, gsl_complex_rect (MaRRe, MaRIm));
                        gsl_matrix_complex_set (MbL, p, q, gsl_complex_rect (MbLRe, MbLIm));
                        gsl_matrix_complex_set (MbR, p, q, gsl_complex_rect (MbRRe, MbRIm));
                        gsl_matrix_complex_set (McL, p, q, gsl_complex_rect (McLRe, McLIm));
                        gsl_matrix_complex_set (McR, p, q, gsl_complex_rect (McRRe, McRIm));
                        gsl_matrix_complex_set (MdL, p, q, gsl_complex_rect (MdLRe, MdLIm));
                        gsl_matrix_complex_set (MdR, p, q, gsl_complex_rect (MdRRe, MdRIm));
                    }
                }
                
                for (p = 0; p <= Ncut-1; p++)
                {
                    for (q = 0; q <= Ncut-1; q++)
                    {
                        Qaab = gsl_matrix_complex_get (Qa, p, q);
                        Qbpab = gsl_matrix_complex_get (Qbp, p, q);
                        Qbmab = gsl_matrix_complex_get (Qbm, p, q);
                        
                        if (p == 0) {
                            
                            Qaamb = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaamb = gsl_matrix_complex_get (Qa, p-1, q);
                        }
                        
                        if (p == Ncut-1) {
                            
                            Qaapb = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaapb = gsl_matrix_complex_get (Qa, p+1, q);
                        }
                        
                        if (q == 0) {
                            
                            Qaabm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaabm = gsl_matrix_complex_get (Qa, p, q-1);
                        }
                        
                        if (q == Ncut-1) {
                            
                            Qaabp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaabp = gsl_matrix_complex_get (Qa, p, q+1);
                        }
                        
                        if (p == 0 || q == 0) {
                            
                            Qaambm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaambm = gsl_matrix_complex_get (Qa, p-1, q-1);
                        }
                        
                        if (p == Ncut-1 || q == Ncut-1) {
                            
                            Qaapbp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaapbp = gsl_matrix_complex_get (Qa, p+1, q+1);
                        }
                        
                        if (p == 0 || q == Ncut-1) {
                            
                            Qaambp = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaambp = gsl_matrix_complex_get (Qa, p-1, q+1);
                        }
                        
                        if (p == Ncut-1 || q == 0) {
                            
                            Qaapbm = gsl_complex_rect (0, 0);
                            
                        } else {
                            
                            Qaapbm = gsl_matrix_complex_get (Qa, p+1, q-1);
                        }
                        
                        MixGaRe = pow(0.5*A1,2) * ( pow(2.0*sin(Kplane),2) * GSL_REAL(Qaab)
                                - 2.0*sin(Kplane) * ( GSL_REAL(Qaamb) - GSL_REAL(Qaapb) + GSL_REAL(Qaabm) - GSL_REAL(Qaabp) )
                                + GSL_REAL(Qaambm) + GSL_REAL(Qaapbp) - GSL_REAL(Qaambp) - GSL_REAL(Qaapbm) );
                        MixGaIm = pow(0.5*A1,2) * ( pow(2.0*sin(Kplane),2) * GSL_IMAG(Qaab)
                                - 2.0*sin(Kplane) * ( GSL_IMAG(Qaamb) - GSL_IMAG(Qaapb) + GSL_IMAG(Qaabm) - GSL_IMAG(Qaabp) )
                                + GSL_IMAG(Qaambm) + GSL_IMAG(Qaapbp) - GSL_IMAG(Qaambp) - GSL_IMAG(Qaapbm) );
                        
                        MixGbRe = pow(A2*sin(Kperp),2) * GSL_REAL(Qbpab);
                        MixGbIm = pow(A2*sin(Kperp),2) * GSL_IMAG(Qbpab);
                        
                        MixGcRe = pow(0.5*A1,2) * ( pow(2.0*sin(Kplane),2) * GSL_REAL(Qaab)
                                - 2.0*sin(Kplane) * ( GSL_REAL(Qaapb) - GSL_REAL(Qaamb) + GSL_REAL(Qaabp) - GSL_REAL(Qaabm) )
                                + GSL_REAL(Qaapbp) + GSL_REAL(Qaambm) - GSL_REAL(Qaapbm) - GSL_REAL(Qaambp) );
                        MixGcIm = pow(0.5*A1,2) * ( pow(2.0*sin(Kplane),2) * GSL_IMAG(Qaab)
                                - 2.0*sin(Kplane) * ( GSL_IMAG(Qaapb) - GSL_IMAG(Qaamb) + GSL_IMAG(Qaabp) - GSL_IMAG(Qaabm) )
                                + GSL_IMAG(Qaapbp) + GSL_IMAG(Qaambm) - GSL_IMAG(Qaapbm) - GSL_IMAG(Qaambp) );
                        
                        MixGdRe = pow(A2*sin(Kperp),2) * GSL_REAL(Qbmab);
                        MixGdIm = pow(A2*sin(Kperp),2) * GSL_IMAG(Qbmab);
                        
                        for (r = 0; r <= Ncut-1; r++)
                        {
                            for (s = 0; s <= Ncut-1; s++)
                            {
                                MaLab = gsl_matrix_complex_get (MaL, p, r);
                                Rapab = gsl_matrix_complex_get (Rap, r, s);
                                MaRab = gsl_matrix_complex_get (MaR, s, q);
                                
                                MixGaRe += ( GSL_REAL(MaLab) * GSL_REAL(Rapab) - GSL_IMAG(MaLab) * GSL_IMAG(Rapab) ) * GSL_REAL(MaRab)
                                        - ( GSL_REAL(MaLab) * GSL_IMAG(Rapab) + GSL_IMAG(MaLab) * GSL_REAL(Rapab) ) * GSL_IMAG(MaRab);
                                
                                MixGaIm += ( GSL_REAL(MaLab) * GSL_REAL(Rapab) - GSL_IMAG(MaLab) * GSL_IMAG(Rapab) ) * GSL_IMAG(MaRab)
                                        + ( GSL_REAL(MaLab) * GSL_IMAG(Rapab) + GSL_IMAG(MaLab) * GSL_REAL(Rapab) ) * GSL_REAL(MaRab);
                                
                                MbLab = gsl_matrix_complex_get (MbL, p, r);
                                Rbpab = gsl_matrix_complex_get (Rbp, r, s);
                                MbRab = gsl_matrix_complex_get (MbR, s, q);
                                
                                MixGbRe += ( GSL_REAL(MbLab) * GSL_REAL(Rbpab) - GSL_IMAG(MbLab) * GSL_IMAG(Rbpab) ) * GSL_REAL(MbRab)
                                        - ( GSL_REAL(MbLab) * GSL_IMAG(Rbpab) + GSL_IMAG(MbLab) * GSL_REAL(Rbpab) ) * GSL_IMAG(MbRab);
                                
                                MixGbIm += ( GSL_REAL(MbLab) * GSL_REAL(Rbpab) - GSL_IMAG(MbLab) * GSL_IMAG(Rbpab) ) * GSL_IMAG(MbRab)
                                        + ( GSL_REAL(MbLab) * GSL_IMAG(Rbpab) + GSL_IMAG(MbLab) * GSL_REAL(Rbpab) ) * GSL_REAL(MbRab);
                                
                                McLab = gsl_matrix_complex_get (McL, p, r);
                                Ramab = gsl_matrix_complex_get (Ram, r, s);
                                McRab = gsl_matrix_complex_get (McR, s, q);
                                
                                MixGcRe += ( GSL_REAL(McLab) * GSL_REAL(Ramab) - GSL_IMAG(McLab) * GSL_IMAG(Ramab) ) * GSL_REAL(McRab)
                                        - ( GSL_REAL(McLab) * GSL_IMAG(Ramab) + GSL_IMAG(McLab) * GSL_REAL(Ramab) ) * GSL_IMAG(McRab);
                                
                                MixGcIm += ( GSL_REAL(McLab) * GSL_REAL(Ramab) - GSL_IMAG(McLab) * GSL_IMAG(Ramab) ) * GSL_IMAG(McRab)
                                        + ( GSL_REAL(McLab) * GSL_IMAG(Ramab) + GSL_IMAG(McLab) * GSL_REAL(Ramab) ) * GSL_REAL(McRab);
                                
                                MdLab = gsl_matrix_complex_get (MdL, p, r);
                                Rbmab = gsl_matrix_complex_get (Rbm, r, s);
                                MdRab = gsl_matrix_complex_get (MdR, s, q);
                                
                                MixGdRe += ( GSL_REAL(MdLab) * GSL_REAL(Rbmab) - GSL_IMAG(MdLab) * GSL_IMAG(Rbmab) ) * GSL_REAL(MdRab)
                                        - ( GSL_REAL(MdLab) * GSL_IMAG(Rbmab) + GSL_IMAG(MdLab) * GSL_REAL(Rbmab) ) * GSL_IMAG(MdRab);
                                
                                MixGdIm += ( GSL_REAL(MdLab) * GSL_REAL(Rbmab) - GSL_IMAG(MdLab) * GSL_IMAG(Rbmab) ) * GSL_IMAG(MdRab)
                                        + ( GSL_REAL(MdLab) * GSL_IMAG(Rbmab) + GSL_IMAG(MdLab) * GSL_REAL(Rbmab) ) * GSL_REAL(MdRab);
                            }
                        }
                        
                        if (p == q) {
                            
                            InvGaRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                    - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 - B1) * (2.0 - cos(Kplane)) - MixGaRe;
                            InvGaIm = 0.5*Gamma - MixGaIm;
                            
                            InvGbRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                    - (C + M) - 2.0*(D2 + B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 + B1) * (2.0 - cos(Kplane)) - MixGbRe;
                            InvGbIm = 0.5*Gamma - MixGbIm;
                            
                            InvGcRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                    - (C - M) - 2.0*(D2 - B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 - B1) * (2.0 - cos(Kplane)) - MixGcRe;
                            InvGcIm = 0.5*Gamma - MixGcIm;
                            
                            InvGdRe = (iw-(Fmax-1.0)/2.0)*Domega + (p-(Ncut-1.0)/2.0)*Bloch
                                    - (C + M) - 2.0*(D2 + B2) * (1.0 - cos(Kperp)) - 2.0*(0.25 + B1) * (2.0 - cos(Kplane)) - MixGdRe;
                            InvGdIm = 0.5*Gamma - MixGdIm;
                            
                        } else if (p == q + 1 || p == q - 1) {
                            
                            InvGaRe = 0.25 - B1 - MixGaRe;
                            InvGaIm = - MixGaIm;
                            
                            InvGbRe = 0.25 + B1 - MixGbRe;
                            InvGbIm = - MixGbIm;
                            
                            InvGcRe = 0.25 - B1 - MixGcRe;
                            InvGcIm = - MixGcIm;
                            
                            InvGdRe = 0.25 + B1 - MixGdRe;
                            InvGdIm = - MixGdIm;
                            
                        } else {
                            
                            InvGaRe = - MixGaRe;
                            InvGaIm = - MixGaIm;
                            
                            InvGbRe = - MixGbRe;
                            InvGbIm = - MixGbIm;
                            
                            InvGcRe = - MixGcRe;
                            InvGcIm = - MixGcIm;
                            
                            InvGdRe = - MixGdRe;
                            InvGdIm = - MixGdIm;
                        }
                        
                        gsl_matrix_complex_set (InvGa, p, q, gsl_complex_rect (InvGaRe, InvGaIm));
                        gsl_matrix_complex_set (InvGb, p, q, gsl_complex_rect (InvGbRe, InvGbIm));
                        gsl_matrix_complex_set (InvGc, p, q, gsl_complex_rect (InvGcRe, InvGcIm));
                        gsl_matrix_complex_set (InvGd, p, q, gsl_complex_rect (InvGdRe, InvGdIm));
                    }
                }
                
                // Free the previous allocation
                
                gsl_matrix_complex_free (Pp);
                gsl_matrix_complex_free (Pm);
                
                gsl_matrix_complex_free (Qa);
                gsl_matrix_complex_free (Qbp);
                gsl_matrix_complex_free (Qbm);
                
                gsl_matrix_complex_free (Rap);
                gsl_matrix_complex_free (Ram);
                gsl_matrix_complex_free (Rbp);
                gsl_matrix_complex_free (Rbm);
                
                gsl_matrix_complex_free (MaL);
                gsl_matrix_complex_free (MaR);
                gsl_matrix_complex_free (MbL);
                gsl_matrix_complex_free (MbR);
                gsl_matrix_complex_free (McL);
                gsl_matrix_complex_free (McR);
                gsl_matrix_complex_free (MdL);
                gsl_matrix_complex_free (MdR);
                
                // Matrix inversion (4 bands)
                
                gsl_permutation *permGa = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Ga = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvGa, permGa, &perm);
                gsl_linalg_complex_LU_invert (InvGa, permGa, Ga);
                
                gsl_permutation *permGb = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Gb = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvGb, permGb, &perm);
                gsl_linalg_complex_LU_invert (InvGb, permGb, Gb);
                
                gsl_permutation *permGc = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Gc = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvGc, permGc, &perm);
                gsl_linalg_complex_LU_invert (InvGc, permGc, Gc);
                
                gsl_permutation *permGd = gsl_permutation_alloc (Ncut);
                gsl_matrix_complex *Gd = gsl_matrix_complex_alloc (Ncut, Ncut);
                gsl_linalg_complex_LU_decomp (InvGd, permGd, &perm);
                gsl_linalg_complex_LU_invert (InvGd, permGd, Gd);
                
                // 4-band retarded local Green's function
                
                gsl_complex kGa = gsl_matrix_complex_get (Ga, n, n);
                gsl_complex kGb = gsl_matrix_complex_get (Gb, n, n);
                gsl_complex kGc = gsl_matrix_complex_get (Gc, n, n);
                gsl_complex kGd = gsl_matrix_complex_get (Gd, n, n);
                
                // Semi-local Dos
                
                OrbA(SemiLocDosSub,iw+Fmax*(n-rank)/Ncpu) = - GSL_IMAG(kGa) / M_PI;
                OrbB(SemiLocDosSub,iw+Fmax*(n-rank)/Ncpu) = - GSL_IMAG(kGb) / M_PI;
                OrbC(SemiLocDosSub,iw+Fmax*(n-rank)/Ncpu) = - GSL_IMAG(kGc) / M_PI;
                OrbD(SemiLocDosSub,iw+Fmax*(n-rank)/Ncpu) = - GSL_IMAG(kGd) / M_PI;
                
                // Free the previous allocation
                
                gsl_permutation_free (permGa);
                gsl_matrix_complex_free (InvGa);
                gsl_matrix_complex_free (Ga);
                
                gsl_permutation_free (permGb);
                gsl_matrix_complex_free (InvGb);
                gsl_matrix_complex_free (Gb);
                
                gsl_permutation_free (permGc);
                gsl_matrix_complex_free (InvGc);
                gsl_matrix_complex_free (Gc);
                
                gsl_permutation_free (permGd);
                gsl_matrix_complex_free (InvGd);
                gsl_matrix_complex_free (Gd);
            }
        }
    }
	
	return SemiLocDosSub;
}
