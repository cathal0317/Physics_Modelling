//
//  Insert list of bugs fixed here.
// Line 5 Added <stdio.h> extension for printf function
// Line 17 Removed unused variable nsteps (redeclared in line 57 anyway with initialisation)
// Line 27 Syntax error for logical OR operator
// Lines 41-44 Memory allocation error fixed with casting to double
// Line 46 Added return 1 for memory allocation error
// Line 50 Initialise ctime to 0.0
// Line 55 Initialise and print the initial state of y and z arrays
// Line 56 Boundary conditions added for readability although they agree with the initial condition
// Lines 62-68 Changed to upwinding scheme instead of FCTC
// Lines 71-72 Copy next values at timestep to y array
// Line 75 Increment time fixed ctime + dt to ctime += dt (Should consider floating point errors)
// Line 86-88 Free memory allocated to y, y_next, z, and z_next
// Line 89 Added return 0 to main function
// Line 97 Added fclose(fptr) for file read error
//

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

long read_input(double *Lx_p, long *nx_p, double *c_p, double *tf_p, double *dt_p, long *ndt_p, double *S_p, char *fname);


int main(void) {
  // **********
  // Parameters
  // **********
  
  long nx, ndt;
  double lx, c, tf, dt, gamma;

  char* fname = "input.txt"; 
  if (read_input(&lx, &nx, &c, &tf, &dt, &ndt, &gamma, fname)) {
    printf("File read error\n");
    return 1;
  }

  double dx = lx/nx;
  if((c*dt/dx > 1.0)||((gamma*dt > 1.0))) {
    printf("Timestep too large\n"); //check if timestep is too large
    return 1;
  }
 
  
  // ************
  // Grid Storage 
  // ************
  double *y, *y_next;
  double *z, *z_next;

  /* Allocate memory according to size of nx */
  y      = (double*) malloc(nx*sizeof(double));
  y_next = (double*) malloc(nx*sizeof(double));
  z      = (double*) malloc(nx*sizeof(double));
  z_next = (double*) malloc(nx*sizeof(double));
  if((y==NULL)||(y_next==NULL)||(z==NULL)||(z_next==NULL)) {
    printf("Allocation error\n");
    return 1;
  }
  
  int j;
  double x;
  double ctime = 0.0;

  // **************
  // initialisation 
  // **************
  // Initialize and print initial state
  for (j = 0; j < nx; j++) {
    double x = j * dx;
    y[j] = exp(-x);
    z[j] = 0.0;
    printf("%g %g %g %g\n", 0.0, x, y[j], z[j]); // printing first because we know the boundary conditions agree with the initial condition
  }

  long ntstep = 0;
  //loop over timesteps 
  while (ctime<tf){

    double yslope, zslope;

    //loop over points 
    y_next[0] = 1.0;  // Boundary condition for y
    z_next[nx-1] = 0.0;  // Boundary condition for z
    
    for (j = 0; j < nx; ++j) {
      // Calculate y_next 
      if (j >= 1) {
        yslope = (y[j] - y[j-1]) / dx;  // flows right (backward)
        y_next[j] = y[j] + dt*( - yslope * c - gamma*y[j]);
      }
      
      // Calculate z_next
      if (j < nx - 1) {
        zslope = (z[j+1] - z[j]) / dx;  // flows left (forward)
        z_next[j] = z[j] + dt*( zslope * c + gamma*y[j]);
      }
    }

    // Copy next values at timestep to y array.
    for (j=0; j<nx; j++) {
      y[j] = y_next[j];
      z[j] = z_next[j]; 
    }

    // Increment time.   
    ctime += dt;
    ntstep++;
    
    // output every ndt
    if (ntstep%ndt==0) {
      for (j=0; j<nx; j++ ) {
	      x = j*dx;
	      printf("%g %g %g %g\n",ctime,x,y[j],z[j]);
      }
    }
  }  

  free(y);
  free(y_next);
  free(z);
  free(z_next);

  return 0;
}

//Reads input parameters, returns an error value (i.e. returns zero on success).
long read_input(double *Lx_p, long *nx_p, double *c_p, double *tf_p, double *dt_p, long *ndt_p, double *S_p, char *fname) {
  FILE* fptr=fopen(fname,"r");
  //Check whether we have successfully opened the file
  if (fptr==NULL) return 1;
  //Check whether we've read correct number of values.
  if (7!=fscanf(fptr,"%lf %ld %lf %lf %lf %ld %lf", Lx_p, nx_p, c_p, tf_p, dt_p, ndt_p, S_p)) {
      fclose(fptr);
      return 1;
  }
  fclose(fptr);
  return 0;
}

