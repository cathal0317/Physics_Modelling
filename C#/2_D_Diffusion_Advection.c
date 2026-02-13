#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl_lapacke.h>

//***********************************************************************************************************************************************************//
//***********************************************************************************************************************************************************//

// ********************
// band_utility.c
// ********************

struct band_mat{
    long ncol;        /* Number of columns in band matrix */
    long nbrows;      /* Number of rows (bands in original matrix) */
    long nbands_up;   /* Number of bands above diagonal */
    long nbands_low;  /* Number of bands below diagonal */
    double *array;    /* Storage for the matrix in banded format */
    /* Internal temporary storage for solving inverse problem */
    long nbrows_inv;  /* Number of rows of inverse matrix */
    double *array_inv;/* Store the inverse if this is generated */
    int *ipiv;        /* Additional inverse information */
  };
  /* Define a new type band_mat */
  typedef struct band_mat band_mat;
  
  /* Initialise a band matrix of a certain size, allocate memory,
     and set the parameters.  */ 
  int init_band_mat(band_mat *bmat, long nbands_lower, long nbands_upper, long n_columns) {
    bmat->nbrows = nbands_lower + nbands_upper + 1;
    bmat->ncol   = n_columns;
    bmat->nbands_up = nbands_upper;
    bmat->nbands_low= nbands_lower;
    bmat->array      = (double *) malloc(sizeof(double)*bmat->nbrows*bmat->ncol);
    bmat->nbrows_inv = bmat->nbands_up*2 + bmat->nbands_low + 1;
    bmat->array_inv  = (double *) malloc(sizeof(double)*(bmat->nbrows+bmat->nbands_low)*bmat->ncol);
    bmat->ipiv       = (int *) malloc(sizeof(int)*bmat->ncol);
    if (bmat->array==NULL||bmat->array_inv==NULL) {
      return 0;
    }  
    /* Initialise array to zero */
    long i;
    for (i=0;i<bmat->nbrows*bmat->ncol;i++) {
      bmat->array[i] = 0.0;
    }
    return 1;
  };
  
  /* Finalise function: should free memory as required */
  void finalise_band_mat(band_mat *bmat) {
    free(bmat->array);
    free(bmat->array_inv);
    free(bmat->ipiv);
  }
  
  /* Get a pointer to a location in the band matrix, using
     the row and column indexes of the full matrix.           */
  double *getp(band_mat *bmat, long row, long column) {
    int bandno = bmat->nbands_up + row - column;
    if(row<0 || column<0 || row>=bmat->ncol || column>=bmat->ncol ) {
      printf("Indexes out of bounds in getp: %ld %ld %ld \n",row,column,bmat->ncol);
      exit(1);
    }
    return &bmat->array[bmat->nbrows*column + bandno];
  }
  
  /* Retrun the value of a location in the band matrix, using
     the row and column indexes of the full matrix.           */
  double getv(band_mat *bmat, long row, long column) {
    return *getp(bmat,row,column);
  }
  
  /* Set an element of a band matrix to a desired value based on the pointer
     to a location in the band matrix, using the row and column indexes
     of the full matrix.           */
  double setv(band_mat *bmat, long row, long column, double val) {
    *getp(bmat,row,column) = val;
    return val;
  }
  
  /* Solve the equation Ax = b for a matrix a stored in band format
     and x and b real arrays                                          */
  int solve_Ax_eq_b(band_mat *bmat, double *x, double *b) {
    /* Copy bmat array into the temporary store */
    int i,bandno;
    for(i=0;i<bmat->ncol;i++) { 
      for (bandno=0;bandno<bmat->nbrows;bandno++) {
        bmat->array_inv[bmat->nbrows_inv*i+(bandno+bmat->nbands_low)] = bmat->array[bmat->nbrows*i+bandno];
      }
      x[i] = b[i];
    }
  
    long nrhs = 1;
    long ldab = bmat->nbands_low*2 + bmat->nbands_up + 1;
    int info = LAPACKE_dgbsv( LAPACK_COL_MAJOR, bmat->ncol, bmat->nbands_low, bmat->nbands_up, nrhs, bmat->array_inv, ldab, bmat->ipiv, x, bmat->ncol);
    return info;
  }
  
  int printmat(band_mat *bmat) {
    long i,j;
    for(i=0; i<bmat->ncol;i++) {
      for(j=0; j<bmat->nbrows; j++) {
        printf("%ld %ld %g \n",i,j,bmat->array[bmat->nbrows*i + j]);
      }
    }
    return 0;
}
  //***********************************************************************************************************************************************************//
  //***********************************************************************************************************************************************************//
  
  // ******************
  // Start of the Code
  // ******************


// *******************************
// Read parameters from input.txt
// *******************************

int read_input(long *nx, long *nv,
               double *t_f, double *Lx,
               double *v_m, double *C,
               long *imin, const char *fname) {
  FILE *fptr = fopen(fname, "r");
  if (fptr == NULL) {
    return 1;
  }
  if (7 != fscanf(fptr, "%ld %ld %lf %lf %lf %lf %ld",
                  nx, nv, t_f, Lx, v_m, C, imin)) {
    fclose(fptr);
    return 1;
  }
  if (*nx <= 2 || *nv <= 2 || *Lx <= 0 || *v_m <= 0 || *imin <= 0) {
    printf("Value error\n");
    fclose(fptr);
    return 1;
  }
  fclose(fptr);
  return 0;
}

// **************************
// Start of the Main Function
// **************************

int main(void) {
  // **********
  // Parameters
  // **********

  double tf, Lx, vm, C;
  long Nx, Nv, Imin;
  const char *fname = "input.txt";

  if (read_input(&Nx, &Nv, &tf, &Lx, &vm, &C, &Imin, fname)) {
    printf("File read error\n");
    return 1;
  }

  // Grid spacing
  double dx = Lx / (Nx - 1);
  double dv = 2.0 * vm / (Nv - 1);

  // Read coefficients F(x) from coefficients.txt
  FILE *fp = fopen("coefficients.txt", "r");
  if (!fp) {
    printf("Error opening coefficients.txt\n");
    return 1;
  }

  double *F = malloc(Nx * sizeof(double));
  if (!F) {
    printf("Memory allocation failed\n");
    fclose(fp);
    return 1;
  }

  for (long i = 0; i < Nx; i++) {
    if (fscanf(fp, "%lf", &F[i]) != 1) {
      printf("Error reading line %ld\n", i + 1);
      fclose(fp);
      free(F);
      return 1;
    }
  }
  fclose(fp);

  // Time step with Imin as a lower bound
  double vmax = vm;
  double Fmax = 0.0;
  for (long i = 0; i < Nx; i++) {
    Fmax = fmax(Fmax, fabs(F[i]));
  }
  double nu = 0.5;
  double dt_cfl = nu / (vmax / dx + Fmax / dv);
  long Nt = (long)ceil(tf / dt_cfl);
  if (Nt < Imin) {
    Nt = Imin;
  }
  double dt = tf / Nt;

  // Total number of points
  long ncols = Nx * Nv;

  // Allocate solution and RHS
  double *f = malloc(ncols * sizeof(double));
  double *f_new = malloc(ncols * sizeof(double));
  double *RHS = malloc(ncols * sizeof(double));

  if (!f || !f_new || !RHS) {
    printf("Memory allocation failed\n");
    free(F);
    free(f);
    free(f_new);
    free(RHS);
    return 1;
  }

  // Initial condition 
  for (long k = 0; k < ncols; k++) {
    f[k] = 0.0;
  }

  // Initialise band matrix
  band_mat bmat;
  long nbands_low = Nv;
  long nbands_up = Nv;
  if (!init_band_mat(&bmat, nbands_low, nbands_up, ncols)) {
    printf("Error: Band matrix initialisation failed\n");
    free(F);
    free(f);
    free(f_new);
    free(RHS);
    return 1;
  }

  // Build constant matrix for implicit Euler
  for (long i = 0; i < Nx; i++) {
    double Fx = F[i];
    for (long j = 0; j < Nv; j++) {
      long idx = i * Nv + j; 
      double vj = j * dv -vm;
      
      int v_boundary = (j == 0 || j == Nv - 1);
      int x_left = (i == 0 && vj >= 0.0);
      int x_right = (i == Nx - 1 && vj <= 0.0);

      if (v_boundary || x_left || x_right) {
        setv(&bmat, idx, idx, 1.0);
        continue;
      }

      double diag = 1.0;

      // Conservative-form upwind fluxes in x and v
      double v_plus = fmax(vj, 0.0);
      double v_minus = fmin(vj, 0.0);
      double F_plus = fmax(Fx, 0.0);
      double F_minus = fmin(Fx, 0.0);

      //***********************************************************************************************************************************************************//

      // Advection
      // x-flux
      diag += dt * (v_plus - v_minus) / dx;
      if (v_plus > 0.0 && i > 0) {
        setv(&bmat, idx, idx - Nv,
             getv(&bmat, idx, idx - Nv) - dt * v_plus / dx);
      }
      if (v_minus < 0.0 && i < Nx - 1) {
        setv(&bmat, idx, idx + Nv,
             getv(&bmat, idx, idx + Nv) + dt * v_minus / dx);
      }

      // v-flux
      diag += dt * (F_plus - F_minus) / dv;
      if (F_plus > 0.0 && j > 0) {
        setv(&bmat, idx, idx - 1,
             getv(&bmat, idx, idx - 1) - dt * F_plus / dv);
      }
      if (F_minus < 0.0 && j < Nv - 1) {
        setv(&bmat, idx, idx + 1,
             getv(&bmat, idx, idx + 1) + dt * F_minus / dv);
      }
      //***********************************************************************************************************************************************************//

      //Diffusion
      // v diffusion
      diag += 2.0 * dt * C / (dv * dv);
      if (j > 0) {
        setv(&bmat, idx, idx - 1, getv(&bmat, idx, idx - 1) - dt * C / (dv * dv));
      }
      if (j < Nv - 1) {
        setv(&bmat, idx, idx + 1, getv(&bmat, idx, idx + 1) - dt * C / (dv * dv));
      }
      setv(&bmat, idx, idx, diag);
      
      //***********************************************************************************************************************************************************//
    }
  }

  // Time stepping
  for (long n = 0; n < Nt; n++) {
    for (long i = 0; i < Nx; i++) {
      for (long j = 0; j < Nv; j++) {
        long idx = i * Nv + j;
        double vj = j * dv -vm;

        int v_boundary = (j == 0 || j == Nv - 1);
        int x_left = (i == 0 && vj >= 0.0);
        int x_right = (i == Nx - 1 && vj <= 0.0);

        if (v_boundary) {
          RHS[idx] = 0.0;
        } else if (x_left || x_right) {
          RHS[idx] = exp(-vj * vj);
        } else {
          RHS[idx] = f[idx];
        }
      }
    }

    int info = solve_Ax_eq_b(&bmat, f_new, RHS);
    if (info != 0) {
      printf("Error: LAPACK solve failed");
      finalise_band_mat(&bmat);
      free(F);
      free(f);
      free(f_new);
      free(RHS);
      return 1;
    }

    // Swap f and f_new
    double *tmp = f;
    f = f_new;
    f_new = tmp;
  }

  // **********************
  // Print out the results
  // **********************

  FILE *out = fopen("output.txt", "w");
  if (!out) {
    printf("Error opening output.txt\n");
    finalise_band_mat(&bmat);
    free(F);
    free(f);
    free(f_new);
    free(RHS);
    return 1;
  }

  for (long j = 0; j < Nv; j++) {
    for (long i = 0; i < Nx; i++) {
      long idx = i * Nv + j;
      fprintf(out, "%g\n", f[idx]);
    }
  }
  fclose(out);

  // Free memory
  finalise_band_mat(&bmat);
  free(F);
  free(f);
  free(f_new);
  free(RHS);

  return 0;
}

  // ****************
  // End of the Code
  // ****************

  //***********************************************************************************************************************************************************//
  //***********************************************************************************************************************************************************//



