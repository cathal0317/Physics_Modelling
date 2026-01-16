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

  long read_input(double *Lx_p, long *nx_p,
                  double *v_p, double *tau_p,
                  double *C_p_p, double *C_q_p,
                  char *fname) {
    FILE *fptr = fopen(fname, "r");
    if (fptr == NULL) {
      return 1;
    }
    if (6 != fscanf(fptr, "%lf %ld %lf %lf %lf %lf",
                    Lx_p, nx_p, v_p, tau_p, C_p_p, C_q_p)) {
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

    double Lx, v, tau, C_p, C_q;
    long nx;

    char *fname = "input.txt";
    if (read_input(&Lx, &nx, &v, &tau, &C_p, &C_q, fname)) {
        printf("File read error\n");
        return 1;
    }

    // Change in x (dx)
    double dx = Lx / (nx - 1);

    // *************************************
    // Read parameters from coefficients.txt
    // *************************************

    // Initialise and read coefficients of D and S
    FILE *fp;
    long ncols = nx;

    double *D   = malloc(ncols * sizeof(double));
    double *S  = malloc(ncols * sizeof(double));  
    // Technically could re-use -S to save memory but for clarity, create separate RHS storage
    double *RHS = malloc(ncols * sizeof(double));   

    if (!D || !S || !RHS) {
        printf("Memory allocation failed\n");
        free(D);
        free(S);
        free(RHS);
        return 1;
    }

    fp = fopen("coefficients.txt", "r");
    if (!fp) {
        printf("Error opening coefficients.txt\n");
        free(D);
        free(S);
        free(RHS);
        return 1;
    }

    for (int i = 0; i < ncols; i++) {
        if (fscanf(fp, "%lf %lf", &D[i], &S[i]) != 2) {
            printf("Error reading line %d\n", i+1);
            fclose(fp);
            free(D);
            free(S);
            free(RHS);
            return 1;
        }
    }

    fclose(fp);

    // Initialise Band Matrix
    band_mat bmat;
    
    long nbands_low = 1;  
    long nbands_up  = 1;
    // Fall back for initialising band matrix
    if (!init_band_mat(&bmat, nbands_low, nbands_up, nx)) {
      printf("Error: Band matrix initialisation failed\n");
      free(D);
      free(S);
      free(RHS);
      return 1;
    }
    
    double *P = malloc(sizeof(double)*ncols);
    double *Q = malloc(sizeof(double)*ncols);

    if (!P || !Q) {
      printf("Memory allocation failed\n");
      finalise_band_mat(&bmat);
      free(D);
      free(S);
      free(RHS);
      free(P);
      free(Q);
      return 1;
    }

    long i;
    
    double dx2 = dx * dx;

    // ************
    // Solve for P
    // ************

    // Left boundary Robin IC

    setv(&bmat, 0, 0, -D[0]/dx - v);
    setv(&bmat, 0, 1,  D[0]/dx);
    RHS[0] = 0.0;

    // Interior points of P
    for(i = 1; i < ncols-1; i++){
      double D_plus_half  = 0.5 * (D[i+1] + D[i]);
      double D_minus_half = 0.5 * (D[i] + D[i-1]);
      
      // Coefficients from conservative upwind scheme
      double P_im1 = D_minus_half/dx2 + v/dx;              
      double P_i   = -(D_plus_half + D_minus_half)/dx2 - v/dx - tau;  
      double P_ip1 = D_plus_half/dx2;                      
      
      setv(&bmat, i, i-1, P_im1);
      setv(&bmat, i, i,   P_i);
      setv(&bmat, i, i+1, P_ip1);

      // Move S to the RHS
      RHS[i] = -S[i];
    }

    // Right boundary Dirichlet BC
    setv(&bmat, ncols-1, ncols-1, 1.0);
    RHS[ncols-1] = C_p;

    //Check whether the solution is valid
    // If not valid, free the memory
    int infoP = solve_Ax_eq_b(&bmat, P, RHS);
    if (infoP != 0) {
      printf("Error: LAPACK solve for P failed with info = %d\n", infoP);

      finalise_band_mat(&bmat);
      free(P);
      free(Q);
      free(D);
      free(S);
      free(RHS);
      return 1;
    }

    // Technically not necessary as we will only rewrite on tridiagonal matrix but for clarity and safety, reset
    for (long j = 0; j < bmat.nbrows * ncols; j++) {
      bmat.array[j] = 0.0;
    }

    // ************
    // Solve for Q
    // ************

    // Left boundary
    setv(&bmat, 0, 0, -D[0]/dx - v);
    setv(&bmat, 0, 1,  D[0]/dx);
    RHS[0] = 0.0;

    // Interior points of Q
    for(i = 1; i < ncols-1; i++){
        double D_plus_half  = 0.5 * (D[i+1] + D[i]);
        double D_minus_half = 0.5 * (D[i] + D[i-1]);
        
        // Coefficients from conservative upwind scheme 
        double Q_im1 = D_minus_half/dx2 + v/dx;              
        double Q_i   = -(D_plus_half + D_minus_half)/dx2 - v/dx;  
        double Q_ip1 = D_plus_half/dx2;                      
        
        setv(&bmat, i, i-1, Q_im1);
        setv(&bmat, i, i,   Q_i);
        setv(&bmat, i, i+1, Q_ip1);

        
        RHS[i] = -tau * P[i];
    }
      
    // Right boundary
    setv(&bmat, ncols-1, ncols-1, 1.0);
    RHS[ncols-1] = C_q;

    //Check whether the solution is valid
    // If not valid, free the memory
    int infoQ = solve_Ax_eq_b(&bmat, Q, RHS);
    if (infoQ != 0) {
      printf("Error: LAPACK solve for Q failed with info = %d\n", infoQ);
      finalise_band_mat(&bmat);
      
      free(D);
      free(S);
      free(RHS);
      free(P);
      free(Q);
      return 1;
    }

    // **********************
    // Print out the results
    // **********************
    
    FILE *out = fopen("output.txt", "w");
    if (!out) {
      printf("Error opening output.txt\n");
      finalise_band_mat(&bmat);
      free(D);
      free(S);
      free(RHS);
      free(P);
      free(Q);
      return 1;
    }

    for(i=0; i<ncols; i++) {
        double x = i * dx;
        fprintf(out, "%g %g %g\n", x, P[i], Q[i]);
    }
    fclose(out);

    // Free the memory allocation
    finalise_band_mat(&bmat);

    free(D);
    free(S);
    free(RHS);
    free(P);
    free(Q);
    
    return 0;
}

  // ****************
  // End of the Code
  // ****************

  //***********************************************************************************************************************************************************//
  //***********************************************************************************************************************************************************//


