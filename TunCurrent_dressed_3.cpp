/**************************************************************/
/* SUBJECT: Dynamically tunable graphene tunneling transistor */
/* COMMENT: Dressed case                                      */
/* METHOD: Keldysh-Floquet Green's function formalism         */
/* PROGRAMMER: Dr. Woo-Ram Lee (wlee51@ua.edu)                */
/**************************************************************/

// DECLARATION: standard library

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// DECLARATION: MPI library

#include <mpi.h>

// DECLARATION: GSL library

#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_eigen.h>

#include <gsl/gsl_rng.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_miser.h>

// INPUT: numerical constants (to be optimized)

int const	    Ncpu       = 100;               // total number of CPUs
int const       MCC        = 20000000/Ncpu;     // number of function calls for Monte Carlo integration (5000000, 20000000)

// INPUT: electron sector (in the unit of Dirac mass)

double const    Ec         = 1.75;              // Dirac band cutoff (4.00)
double const    Eta        = 0.01;              // level broadening width
double const    Temp       = 0.02;              // temperature

// INPUT: photon sector

double const    Field      = 0.30;              // optical field strength
double const    Freq       = 1.25;              // optical frequency
double const    Pol        = 0.50*M_PI;         // optical polarization (0: linear, 0.5*M_PI: circular)

// INPUT: tuning parameter

int const       Bias_cut   = 25;                // grid number of bias (150)
double const    Bias_init  = 3.50;              // grid size of bias
double const    Bias_diff  = 0.02;              // grid size of bias

//  75, 0.00
//  75, 1.50
//  25, 3.50
//  10, 2.39
//  15, 3.58


/**********************************************************************/
/* SUBLOUTINE: Determination of Floquet mode cutoff                   */
/**********************************************************************/

int Subroutine_Shift (double Ek)
{
    // DECLARATION: variables and arrays
    
    int r, s, p, q, n, n_init, n_ref;
    double omega, Bias, Dev;
    
    // INPUT: Initial guess of the Floquet mode cutoff
    
    // STEP 0: Determination of the Floquet mode cuoff
    
    Bias = 0;
    omega = 1.0 + 0.5 * 1.0 * Bias;
    
    r = 0;
    
    n_init = 1 + 2*ceil(2.0/Freq);
    
    // LOOP: Floquet mode
    
    for (n = n_init; n <= 999999; n++)
    {
        // INPUT: Inverse retarded Green's function
        
        gsl_permutation *perm = gsl_permutation_alloc (n);
        gsl_matrix_complex *InvGr = gsl_matrix_complex_alloc (n, n);
        gsl_matrix_complex *Gr_Ta = gsl_matrix_complex_alloc (n, n);
        
        for (p = 0; p <= n-1; p++)
        {
            for (q = 0; q <= n-1; q++)
            {
                if (p == q) {
                    
                    gsl_matrix_complex_set (InvGr, p, q,
                                            gsl_complex_sub (
                                                             gsl_complex_sub (
                                                                              gsl_complex_rect (omega + (p-(n-1)/2) * Freq - 0.5 * (1.0 + 1.0 * Bias), Eta),
                                                                              gsl_complex_mul_real (gsl_complex_inverse (gsl_complex_rect (omega + (p-(n-1)/2) * Freq - 0.5 * (- 1.0 + 1.0 * Bias), Eta)), pow(1.0,2) - pow(0.5,2))
                                                                              ),
                                                             gsl_complex_add (
                                                                              gsl_complex_mul_real (
                                                                                                    gsl_complex_inverse (gsl_complex_rect (omega + (p-1-(n-1)/2) * Freq - 0.5 * (- 1.0 + 1.0 * Bias), Eta)),
                                                                                                    2.0 * pow(0.5 * Field / Freq,2) * (1.0 - 1.0 * 1.0 * sin(Pol))
                                                                                                    ),
                                                                              gsl_complex_mul_real (
                                                                                                    gsl_complex_inverse (gsl_complex_rect (omega + (p+1-(n-1)/2) * Freq - 0.5 * (- 1.0 + 1.0 * Bias), Eta)),
                                                                                                    2.0 * pow(0.5 * Field / Freq,2) * (1.0 + 1.0 * 1.0 * sin(Pol))
                                                                                                    )
                                                                              )
                                                             )
                                            );
                    
                } else if (abs(p-q) == 1) {
                    
                    gsl_matrix_complex_set (InvGr, p, q,
                                            gsl_complex_add (
                                                             gsl_complex_div (
                                                                              gsl_complex_mul (
                                                                                               gsl_complex_polar (- (p-q) * 1.0 * 0.5 * Field / Freq * sqrt(pow(1.0,2) - pow(0.5,2)), - 1.0 * 1.0 * 0),
                                                                                               gsl_complex_rect (- 1.0 * cos((p-q)*Pol), 1.0 + 1.0 * sin((p-q)*Pol))
                                                                                               ),
                                                                              gsl_complex_rect (omega + (p-(n-1)/2) * Freq - 0.5 * (- 1.0 + 1.0 * Bias), Eta)
                                                                              ),
                                                             gsl_complex_div (
                                                                              gsl_complex_mul (
                                                                                               gsl_complex_polar (- (p-q) * 1.0 * 0.5 * Field / Freq * sqrt(pow(1.0,2) - pow(0.5,2)), 1.0 * 1.0 * 0),
                                                                                               gsl_complex_rect (1.0 * cos((p-q)*Pol), 1.0 - 1.0 * sin((p-q)*Pol))
                                                                                               ),
                                                                              gsl_complex_rect (omega + (q-(n-1)/2) * Freq - 0.5 * (- 1.0 + 1.0 * Bias), Eta)
                                                                              )
                                                             )
                                            );
                    
                } else if (p == q + 2) {
                    
                    gsl_matrix_complex_set (InvGr, p, q,
                                            gsl_complex_div (
                                                             gsl_complex_mul_real (gsl_complex_rect (1.0 + cos(2.0*Pol), - sin(2.0*Pol)), pow(0.5*Field/Freq,2)),
                                                             gsl_complex_rect (omega + (p-1-(n-1)/2) * Freq - 0.5 * (- 1.0 + 1.0 * Bias), Eta)
                                                             )
                                            );
                    
                } else if (p == q - 2) {
                    
                    gsl_matrix_complex_set (InvGr, p, q,
                                            gsl_complex_div (
                                                             gsl_complex_mul_real (gsl_complex_rect (1.0 + cos(2.0*Pol), sin(2.0*Pol)), pow(0.5*Field/Freq,2)),
                                                             gsl_complex_rect (omega + (p+1-(n-1)/2) * Freq - 0.5 * (- 1.0 + 1.0 * Bias), Eta)
                                                             )
                                            );
                    
                } else {
                    
                    gsl_matrix_complex_set (InvGr, p, q, gsl_complex_rect (0, 0));
                }
            }
        }
        
        // CALCULATION: matrix inversion to get the retarded Green's function
        
        gsl_linalg_complex_LU_decomp (InvGr, perm, &s);
        gsl_linalg_complex_LU_invert (InvGr, perm, Gr_Ta);
        
        // CRITERION: matching with the Green's function at the boundary Floquet mode - Part 1
        
        Dev = fabs(GSL_IMAG (gsl_matrix_complex_get (Gr_Ta, 0, 0)) / GSL_IMAG (gsl_matrix_complex_get (Gr_Ta, (n-1)/2, (n-1)/2)));
        
        //        printf ("Ncut = %d, ImGr = %e, Dev = %e\n", n, GSL_IMAG (gsl_matrix_complex_get (Gr_Ta, n-1, n-1)), Dev);
        
        if (Dev < 0.01) {
            
            r += 1;
            
        } else {
            
            r = 0;
        }
        
        // FREE: previous allocation for arrays
        
        gsl_permutation_free (perm);
        gsl_matrix_complex_free (InvGr);
        gsl_matrix_complex_free (Gr_Ta);
        
        // CRITERION: matching with the equilibrium Green's function at the boundary - Part 2
        
        if (r >= 3) {
            
            n_ref = n;
            
            break;
        }
    }
    
    return n_ref;
}


/**********************************************************************/
/* SUBLOUTINE: Integrand in the inter-layer tunneling current formula */
/**********************************************************************/

struct gsl_params
{
    double Bias;
};

double f_Integrand (double *vars, size_t dim, void *params)
{
    // DECLARATION: variables for function
    
    // vars[0]: variable for frequency domain
    // vars[1]: variable for energy dispersion
    // vars[2]: variable for polar angle in 2D momentum space
    
    // DECLARATION: parameters for function
    
    double Bias = ((struct gsl_params *) params) -> Bias;
    
    // DECLARATION: variables
    
    int sgn_val, sgn_band, sgn_lay, sgn_lat;
    int p, q, n, s, n_ref, Ncut;
    double Mu_T, Mu_B;
    double FDOS_T, FDOS_B;
    double DOS_T, DOS_B;
    double Dist_T, Dist_B;
    double Integrand;
    
    // INPUT: Chemical potentials
    // ASSUMPTION: half-filling
    
    Mu_T = 0.5*Bias;
    Mu_B = -0.5*Bias;
    
    // CALCULATION: Determination of Floquet mode cutoff
    
    n_ref = Subroutine_Shift (vars[1]);
    Ncut = n_ref + 4*ceil(vars[1]);
    
    // CALCULATION: Summation over discrete variables (valley pseudospin, band index, Floquet mode)
    
    Integrand = 0;
    
    for (sgn_val = -1; sgn_val <= 1; sgn_val += 2)  // LOOP: Valley (1 for K point; -1 for K' point)
    {
        for (sgn_band = -1; sgn_band <= 1; sgn_band += 2)   // LOOP: Band (1 for conduction; -1 for valence)
        {
            gsl_matrix_complex *Gr_Ta = gsl_matrix_complex_alloc (Ncut, Ncut);
            gsl_matrix_complex *Gr_Tb = gsl_matrix_complex_alloc (Ncut, Ncut);
            gsl_matrix_complex *Gr_Ba = gsl_matrix_complex_alloc (Ncut, Ncut);
            gsl_matrix_complex *Gr_Bb = gsl_matrix_complex_alloc (Ncut, Ncut);
            
            for (sgn_lay = -1; sgn_lay <= 1; sgn_lay += 2)  // LOOP: layer (1 for top; -1 for bottom)
            {
                for (sgn_lat = -1; sgn_lat <= 1; sgn_lat += 2)  // LOOP: sublattice (1 for "a"; -1 for "b")
                {
                    // INPUT: Inverse retarded Green's function
                    
                    gsl_matrix_complex *InvGr = gsl_matrix_complex_alloc (Ncut, Ncut);
                    gsl_permutation *perm = gsl_permutation_alloc (Ncut);
                    
                    for (p = 0; p <= Ncut-1; p++)
                    {
                        for (q = 0; q <= Ncut-1; q++)
                        {
                            if (p == q) {
                                
                                gsl_matrix_complex_set (InvGr, p, q,
                                                        gsl_complex_sub (
                                                                         gsl_complex_sub (
                                                                                          gsl_complex_rect (vars[0] + (p-(Ncut-1)/2) * Freq - 0.5 * (sgn_lat + sgn_lay * Bias), Eta),
                                                                                          gsl_complex_mul_real (gsl_complex_inverse (gsl_complex_rect (vars[0] + (p-(Ncut-1)/2) * Freq - 0.5 * (- sgn_lat + sgn_lay * Bias), Eta)), pow(vars[1],2) - pow(0.5,2))
                                                                                          ),
                                                                         gsl_complex_add (
                                                                                          gsl_complex_mul_real (
                                                                                                                gsl_complex_inverse (gsl_complex_rect (vars[0] + (p-1-(Ncut-1)/2) * Freq - 0.5 * (- sgn_lat + sgn_lay * Bias), Eta)),
                                                                                                                2.0 * pow(0.5 * Field / Freq,2) * (1.0 - sgn_lat * sgn_val * sin(Pol))
                                                                                                                ),
                                                                                          gsl_complex_mul_real (
                                                                                                                gsl_complex_inverse (gsl_complex_rect (vars[0] + (p+1-(Ncut-1)/2) * Freq - 0.5 * (- sgn_lat + sgn_lay * Bias), Eta)),
                                                                                                                2.0 * pow(0.5 * Field / Freq,2) * (1.0 + sgn_lat * sgn_val * sin(Pol))
                                                                                                                )
                                                                                          )
                                                                         )
                                                        );
                                
                            } else if (abs(p-q) == 1) {
                                
                                gsl_matrix_complex_set (InvGr, p, q,
                                                        gsl_complex_add (
                                                                         gsl_complex_div (
                                                                                          gsl_complex_mul (
                                                                                                           gsl_complex_polar (- (p-q) * sgn_val * 0.5 * Field / Freq * sqrt(pow(vars[1],2) - pow(0.5,2)), - sgn_lat * sgn_val * vars[2]),
                                                                                                           gsl_complex_rect (- sgn_lat * cos((p-q)*Pol), sgn_val + sgn_lat * sin((p-q)*Pol))
                                                                                                           ),
                                                                                          gsl_complex_rect (vars[0] + (p-(Ncut-1)/2) * Freq - 0.5 * (- sgn_lat + sgn_lay * Bias), Eta)
                                                                                          ),
                                                                         gsl_complex_div (
                                                                                          gsl_complex_mul (
                                                                                                           gsl_complex_polar (- (p-q) * sgn_val * 0.5 * Field / Freq * sqrt(pow(vars[1],2) - pow(0.5,2)), sgn_lat * sgn_val * vars[2]),
                                                                                                           gsl_complex_rect (sgn_lat * cos((p-q)*Pol), sgn_val - sgn_lat * sin((p-q)*Pol))
                                                                                                           ),
                                                                                          gsl_complex_rect (vars[0] + (q-(Ncut-1)/2) * Freq - 0.5 * (- sgn_lat + sgn_lay * Bias), Eta)
                                                                                          )
                                                                         )
                                                        );
                                
                            } else if (p == q + 2) {
                                
                                gsl_matrix_complex_set (InvGr, p, q,
                                                        gsl_complex_div (
                                                                         gsl_complex_mul_real (gsl_complex_rect (1.0 + cos(2.0*Pol), - sin(2.0*Pol)), pow(0.5*Field/Freq,2)),
                                                                         gsl_complex_rect (vars[0] + (p-1-(Ncut-1)/2) * Freq - 0.5 * (- sgn_lat + sgn_lay * Bias), Eta)
                                                                         )
                                                        );
                                
                            } else if (p == q - 2) {
                                
                                gsl_matrix_complex_set (InvGr, p, q,
                                                        gsl_complex_div (
                                                                         gsl_complex_mul_real (gsl_complex_rect (1.0 + cos(2.0*Pol), sin(2.0*Pol)), pow(0.5*Field/Freq,2)),
                                                                         gsl_complex_rect (vars[0] + (p+1-(Ncut-1)/2) * Freq - 0.5 * (- sgn_lat + sgn_lay * Bias), Eta)
                                                                         )
                                                        );
                                
                            } else {
                                
                                gsl_matrix_complex_set (InvGr, p, q, gsl_complex_rect (0, 0));
                            }
                        }
                    }
                    
                    // CALCULATION: matrix inversion to get the retarded Green's function
                    
                    gsl_linalg_complex_LU_decomp (InvGr, perm, &s);
                    
                    if (sgn_lay == 1 && sgn_lat == 1) {
                        
                        gsl_linalg_complex_LU_invert (InvGr, perm, Gr_Ta);
                    }
                    
                    if (sgn_lay == 1 && sgn_lat == -1) {
                        
                        gsl_linalg_complex_LU_invert (InvGr, perm, Gr_Tb);
                    }
                    
                    if (sgn_lay == -1 && sgn_lat == 1) {
                        
                        gsl_linalg_complex_LU_invert (InvGr, perm, Gr_Ba);
                    }
                    
                    if (sgn_lay == -1 && sgn_lat == -1) {
                        
                        gsl_linalg_complex_LU_invert (InvGr, perm, Gr_Bb);
                    }
                    
                    // FREE: previous allocation for arrays
                    
                    gsl_matrix_complex_free (InvGr);
                    gsl_permutation_free (perm);
                }
            }
            
            for (n = 1; n <= Ncut-2; n++)
            {
                // CALCULATION: Photon-dressed conversion factor
                
                gsl_complex Gr_b_p0_n1;
                gsl_complex Gr_b_p0_n_1;
                gsl_complex Gr_a_p0_n1;
                gsl_complex Gr_a_p0_n_1;
                gsl_complex Gr_a_p1_n1;
                gsl_complex Gr_a_p_1_n_1;
                gsl_complex Gr_a_p1_n_1;
                gsl_complex Gr_a_p_1_n1;
                gsl_complex U_a, U_b, Ph_U;
                
                gsl_vector_complex *U_T = gsl_vector_complex_alloc (Ncut);
                gsl_vector_complex *U_B = gsl_vector_complex_alloc (Ncut);
                
                for (p = 1; p <= Ncut-2; p++)
                {
                    if (p == n) {
                        
                        s = 1;
                        
                    } else {
                        
                        s = 0;
                    }
                    
                    if (sgn_band == 1) {
                        
                        U_a = gsl_complex_rect (cos(0.5*acos(0.5/vars[1])), 0);
                        U_b = gsl_complex_polar (sgn_val * sin(0.5*acos(0.5/vars[1])), sgn_val * vars[2]);
                    }
                    
                    if (sgn_band == -1) {
                        
                        U_a = gsl_complex_polar (sgn_val * sin(0.5*acos(0.5/vars[1])), - sgn_val * vars[2]);
                        U_b = gsl_complex_rect (- cos(0.5*acos(0.5/vars[1])), 0);
                    }
                    
                    for (sgn_lay = -1; sgn_lay <= 1; sgn_lay += 2)  // LOOP: layer (1 for top; -1 for bottom)
                    {
                        if (sgn_lay == 1) {
                            
                            Gr_b_p0_n1 = gsl_matrix_complex_get (Gr_Tb, p, n+1);
                            Gr_b_p0_n_1 = gsl_matrix_complex_get (Gr_Tb, p, n-1);
                            Gr_a_p0_n1 = gsl_matrix_complex_get (Gr_Ta, p, n+1);
                            Gr_a_p0_n_1 = gsl_matrix_complex_get (Gr_Ta, p, n-1);
                            Gr_a_p1_n1 = gsl_matrix_complex_get (Gr_Ta, p+1, n+1);
                            Gr_a_p_1_n_1 = gsl_matrix_complex_get (Gr_Ta, p-1, n-1);
                            Gr_a_p1_n_1 = gsl_matrix_complex_get (Gr_Ta, p+1, n-1);
                            Gr_a_p_1_n1 = gsl_matrix_complex_get (Gr_Ta, p-1, n+1);
                        }
                        
                        if (sgn_lay == -1) {
                            
                            Gr_b_p0_n1 = gsl_matrix_complex_get (Gr_Bb, p, n+1);
                            Gr_b_p0_n_1 = gsl_matrix_complex_get (Gr_Bb, p, n-1);
                            Gr_a_p0_n1 = gsl_matrix_complex_get (Gr_Ba, p, n+1);
                            Gr_a_p0_n_1 = gsl_matrix_complex_get (Gr_Ba, p, n-1);
                            Gr_a_p1_n1 = gsl_matrix_complex_get (Gr_Ba, p+1, n+1);
                            Gr_a_p_1_n_1 = gsl_matrix_complex_get (Gr_Ba, p-1, n-1);
                            Gr_a_p1_n_1 = gsl_matrix_complex_get (Gr_Ba, p+1, n-1);
                            Gr_a_p_1_n1 = gsl_matrix_complex_get (Gr_Ba, p-1, n+1);
                        }
                        
                        Ph_U = gsl_complex_add (
                                                gsl_complex_add (
                                                                 gsl_complex_mul_real (U_b, s),
                                                                 gsl_complex_mul (
                                                                                  gsl_complex_mul_real (U_a, 0.5*Field/Freq),
                                                                                  gsl_complex_add (
                                                                                                   gsl_complex_mul (gsl_complex_rect (- cos(Pol), sgn_val + sin(Pol)), Gr_b_p0_n1),
                                                                                                   gsl_complex_mul (gsl_complex_rect (cos(Pol), - sgn_val + sin(Pol)), Gr_b_p0_n_1)
                                                                                                   )
                                                                                  )
                                                                 ),
                                                gsl_complex_mul (
                                                                 gsl_complex_div (U_b, gsl_complex_rect (vars[0] + (p-(Ncut-1)/2) * Freq - 0.5 * (- 1.0 + sgn_lay * Bias), Eta)),
                                                                 gsl_complex_add (
                                                                                  gsl_complex_mul (
                                                                                                   gsl_complex_polar (0.5*Field/Freq * sqrt(pow(vars[1],2) - pow(0.5,2)) * sgn_val, sgn_val * vars[2]),
                                                                                                   gsl_complex_add (
                                                                                                                    gsl_complex_mul (gsl_complex_rect (cos(Pol), sgn_val - sin(Pol)), Gr_a_p0_n1),
                                                                                                                    gsl_complex_mul (gsl_complex_rect (- cos(Pol), - sgn_val - sin(Pol)), Gr_a_p0_n_1)
                                                                                                                    )
                                                                                                   ),
                                                                                  gsl_complex_mul_real (
                                                                                                        gsl_complex_sub (
                                                                                                                         gsl_complex_add (
                                                                                                                                          gsl_complex_mul_real (Gr_a_p1_n1, 2.0*(1.0 - sgn_val * sin(Pol))),
                                                                                                                                          gsl_complex_mul_real (Gr_a_p_1_n_1, 2.0*(1.0 + sgn_val * sin(Pol)))
                                                                                                                                          ),
                                                                                                                         gsl_complex_add (
                                                                                                                                          gsl_complex_mul (gsl_complex_rect (1.0 + cos(2.0*Pol), sin(2.0*Pol)), Gr_a_p1_n_1),
                                                                                                                                          gsl_complex_mul (gsl_complex_rect (1.0 + cos(2.0*Pol), - sin(2.0*Pol)), Gr_a_p_1_n1)
                                                                                                                                          )
                                                                                                                         ),
                                                                                                        pow(0.5*Field/Freq,2)
                                                                                                        )
                                                                                  )
                                                                 )
                                                );
                        
                        if (sgn_lay == 1) {
                            
                            gsl_vector_complex_set (U_T, p, Ph_U);
                        }
                        
                        if (sgn_lay == -1) {
                            
                            gsl_vector_complex_set (U_B, p, Ph_U);
                        }
                    }
                }
                
                // CALCULATION: Photon-dressed spectral density
                
                FDOS_T = FDOS_B = 0;
                
                for (p = 1; p <= Ncut-2; p++)
                {
                    for (q = 1; q <= Ncut-2; q++)
                    {
                        FDOS_T += - GSL_IMAG (gsl_complex_mul (gsl_complex_mul (gsl_complex_conjugate (gsl_vector_complex_get (U_B, p)), gsl_matrix_complex_get (Gr_Tb, p, q)), gsl_vector_complex_get (U_B, q))) / M_PI;
                        FDOS_B += - GSL_IMAG (gsl_complex_mul (gsl_complex_mul (gsl_complex_conjugate (gsl_vector_complex_get (U_T, p)), gsl_matrix_complex_get (Gr_Bb, p, q)), gsl_vector_complex_get (U_T, q))) / M_PI;
                    }
                }
                
                // CALCULATION: Photon-undressed spectral density & distribution
                
                DOS_T = - GSL_IMAG (gsl_complex_inverse (gsl_complex_rect (vars[0] + (n-(Ncut-1)/2)*Freq - sgn_band * vars[1] - Mu_T, Eta))) / M_PI;
                DOS_B = - GSL_IMAG (gsl_complex_inverse (gsl_complex_rect (vars[0] + (n-(Ncut-1)/2)*Freq - sgn_band * vars[1] - Mu_B, Eta))) / M_PI;
                
                Dist_T = 1.0 / (exp((vars[0] + (n-(Ncut-1)/2)*Freq - Mu_T)/Temp) + 1.0);
                Dist_B = 1.0 / (exp((vars[0] + (n-(Ncut-1)/2)*Freq - Mu_B)/Temp) + 1.0);
                
                // CALCULATION: Integrand in the inter-layer tunneling current formula
                
                Integrand += vars[1] * (FDOS_B * DOS_T * Dist_T - FDOS_T * DOS_B * Dist_B);
                
                // FREE: previous allocation for arrays
                
                gsl_vector_complex_free (U_T);
                gsl_vector_complex_free (U_B);
            }
            
            // FREE: previous allocation for arrays
            
            gsl_matrix_complex_free (Gr_Ta);
            gsl_matrix_complex_free (Gr_Tb);
            gsl_matrix_complex_free (Gr_Ba);
            gsl_matrix_complex_free (Gr_Bb);
        }
    }
    
    return Integrand;
}


/*************************************************************/
/* SUBLOUTINE: Inter-layer tunneling current vs bias voltage */
/*************************************************************/

double Subroutine_TunCurrent (int rank, double Bias)
{
    // DECLARATION: variables and arrays
    
    double TunCurrent_Sub;
    double error;
    
    // CALCULATION: Determination of Floquet mode cutoff
    
    double xl[3] = {-0.5*Freq, 0.5, 0};
    double xu[3] = {0.5*Freq, Ec, 2.0*M_PI};
    
    // INTEGRATION: Monte Carlo Method (MISER)
    
    struct gsl_params params_p = { Bias };
    gsl_monte_function F_Integrand = { &f_Integrand, 3, &params_p };
    
    const gsl_rng_type *T;
    gsl_rng *r;
    gsl_rng_env_setup ();
    
    size_t calls = MCC;
    
    T = gsl_rng_default;
    r = gsl_rng_alloc (T);
    
    // Prescription for MPI
    
    gsl_rng_set (r, rank + 1);
    
    {
        gsl_monte_miser_state *s = gsl_monte_miser_alloc (3);
        gsl_monte_miser_integrate (&F_Integrand, xl, xu, 3, calls, r, s, &TunCurrent_Sub, &error);
        gsl_monte_miser_free (s);
    }
    
    gsl_rng_free (r);
    
    return TunCurrent_Sub;
}


/********************************************************/
/*                      MAIN ROUTINE                    */
/********************************************************/

main(int argc, char **argv)
{
    // OPEN: saving files
    
//    FILE *f1;
//    f1 = fopen("TunCurrent_Ec04p0_T0p02_Eta0p01_E0p10_F1p50_P0p50pi_Mc5000000","wt");
    
    // INITIALIZATION: MPI
    
    int rank;       // Rank of my CPU
    int dest = 0;	// CPU receiving message
    int tag = 0;	// Indicator distinguishing messages
    
    MPI_Status status;
    MPI_Init (&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    
    // DECLARATION: variables and arrays
    
    int j;
    double Bias;
    double TunCurrent_Sub;
    double TunCurrent;
    
    // CALCULATION: Inter-layer tunneling current
    
    // SAVE: {bias voltage, inter-layer tunneling current}
    
    if (rank == 0) {
        
        printf("%f %e\n", 0.0, 0.0);
//        fprintf(f1, "%f %e\n", 0.0, 0.0);
    }
    
    for (j = 1; j <= Bias_cut; j++)     // LOOP: Bias voltage
    {
        Bias = Bias_init + j * Bias_diff;
        
        // PARALLELIZING: MPI
        
        TunCurrent_Sub = Subroutine_TunCurrent (rank, Bias);
        
        MPI_Reduce (&TunCurrent_Sub, &TunCurrent, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        // SAVE: {bias voltage, inter-layer tunneling current}
        
        if (rank == 0) {
            
            printf("%f %e\n", Bias, TunCurrent/Ncpu);
//            fprintf(f1, "%f %e\n", Bias, TunCurrent/Ncpu);
        }
    }
    
    if (rank == 0) {
        
        printf("\n");
//        fprintf(f1, "\n");
    }
    
    // CLOSE: saving files
    
//    fclose(f1);
    
    // FINISH: MPI
    
    MPI_Finalize();
}

