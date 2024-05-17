/************************************************************************************************/
/*********************** [System] Electric field-driven hypercubic lattice **********************/
/*********************** [Method] Floquet-Keldysh DMFT                     **********************/
/*********************** [Object] Local dos/ocupation/distribution         **********************/
/*********************** [Programmer] Dr. Woo-Ram Lee (wrlee@kias.re.kr)   **********************/
/************************************************************************************************/

// Standard libraries

#include <stdio.h>
#include <math.h>

// GSL libraries

#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_sf_bessel.h>

int const		Iband  = 25*45;
int const		Icut   = 25*1155;
int const		Ncut   = 1000001;
double const	Domega = 1.0/Iband;
double const	Eta    = 5.0*Domega;

int main()
{	
	// Open the saving file
	
	FILE *f1;
	f1 = fopen("U0.0","wt");
	
	// Define local variables
	
	int		Fmax, i, iw, n;
	double	Field, LocDos[Icut];

	// Loop for changing field strength E
	
	for (i = 0; i <= 158; i++)
	{
		if (i == 0) {
			
			// Local DOS in equilibrium
			
			for (iw = 0; iw <= Icut-1; iw++)
			{
				LocDos[iw] = exp(-pow((iw-(Icut-1.0)/2.0)*Domega,2)) / sqrt(M_PI);
			}
			
		} else {
			
			// Input strength of electric field E
			
			Fmax  = 25*(2*i-1);
			Field = 1.0*Fmax/Iband;
			
			// Local DOS out of equilibrium
			
			for (iw = 0; iw <= Icut-1; iw++)
			{
				LocDos[iw] = gsl_sf_bessel_In_scaled(0, 0.5/pow(Field,2)) 
							 * Eta/(pow((iw-(Icut-1.0)/2.0)*Domega,2)+pow(Eta,2))/M_PI;
				
				for (n = 1; n <= (Ncut-1)/2; n++)
				{
					if (gsl_sf_bessel_In_scaled(n, 0.5/pow(Field,2)) > 0.001) {
						
						LocDos[iw] += gsl_sf_bessel_In_scaled(n, 0.5/pow(Field,2))
									  *(Eta/(pow((iw-(Icut-1.0)/2.0)*Domega + n*Field,2)+pow(Eta,2))/M_PI
									  + Eta/(pow((iw-(Icut-1.0)/2.0)*Domega - n*Field,2)+pow(Eta,2))/M_PI);
						
					} else {
						
						break;
					}
				}
			}
		}
		
		// Print out status
		
		printf ("Field = %f \n", Field);
		
		// Save data on output file
		
		for (iw = (Icut-1)/2 - 10*Iband; iw <= (Icut-1)/2 + 10*Iband; iw++)
		{
			fprintf(f1, "%f %f %f \n", Field, (iw-(Icut-1.0)/2.0)*Domega, LocDos[iw]);
		}
		
		fprintf(f1, " \n");
	}
	
	// Close the saving file
	
	fclose(f1);

	return 0;
}
