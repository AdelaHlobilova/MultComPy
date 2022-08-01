/*
    Lineal path function computed by brute force.

    Lineal path function is a probability that a line segment lies in the same
    phase when randomly thrown into the microstructure [1]. First, all the
    paths are generated (by function L2_generate_paths). Since the lineal path
    is point symmetric [2], only a half of the paths is generated and the
    second half of the lineal path function is obtained thanks to the symmetry.
    I.e. the algorithm generates all paths from 0 degrees to 180 degrees with
    different lengths (from length = min(row, columns) to
                       length = number of image max(rows, columns)).
    Only the longest possible paths are generated and the shorter ones are
    omitted since they are already contained in the longer ones. The less
    accuracy is since the shorter paths generated by Bressenham's line
    algorithm with the same angle may differ in some pixels. This algorithm is
    inspired by the Torquato algorithm (Random heterogeneous materials [1],
    pp. 291). The Lineal path algorithm translates all paths along with the
    medium.

    USAGE ON LINUX MACHINES:
    compile using gcc to create shared object as follows:
    gcc -std=c11 -Wall -Wextra -pedantic -c -fPIC -O3 -flto -ftree-loop-optimize -ftree-loop-vectorize lineal_path_c.c -o lineal_path_c.o
    gcc -shared lineal_path_c.o -o lineal_path_c.so

    USAGE ON WINDOWS MACHINES:
    gcc -std=c11 -Wall -Wextra -pedantic -c -fPIC lineal_path_c.c -o lineal_path_c.o
    gcc -shared lineal_path_c.o -o lineal_path_c.dll

    References
    ----------
    [1] Torquato, S. (2002). Random Heterogeneous Materials:
    Microstructure and Macroscopic Properties. Interdisciplinary Applied
    Mathematics, Springer, New York, NY, 703 pages. Pp. 285.
    [2] Havelka, J., Ku�erov�, A., & S�kora, J. (2016). Compression and
    reconstruction of random microstructures using accelerated lineal path
    function. Computational Materials Science, 122, 102-117.

    @author:   Adela Hlobilova, adela.hlobilova@gmail.com
    Last edit on May 17, 2022
*/

#include <stdio.h>
#include <stdlib.h>
#include<string.h>
#include<math.h>
#include <time.h>

int offset(int x, int y, int z, int row, int col);
int largest(int arr[], int n);
void Bresenham3D(int x1, int y1, int z1, int x2, int y2, int z2, int* path, int *steps);
void L2_generate_paths(int dep, int row, int col, int imgdep, int imgrow, int imgcol, int step_in_image, int** paths, int* count_vox_in_paths, int *len_paths, int *num_paths);  //, int *len_paths, int *paths
void L2_direct_computation(int *A, int dep, int row, int col, int phase, int** paths, int* count_vox_in_paths, int *len_paths, int num_paths, double* L2_mat_prob, int progress_flag, int start_dep, int stop_dep);
void Lineal_path(int dep, int row, int col, int imgdep, int imgrow, int imgcol, int phase, int step_in_image, int *img_array, double *L2_mat_prob, int progress_flag, int start_dep, int stop_dep);


int offset(int x, int y, int z, int row, int col) {
    return x*row*col + y*col + z;
}


int largest(int arr[], int n) {
    int i;
    int max = arr[0];

    for (i = 1; i < n; i++)
        if (arr[i] > max)
            max = arr[i];
    return max;
}


void Bresenham3D(int x1, int y1, int z1, int x2, int y2, int z2, int* path, int *steps){

    int dx, dy, dz, xs, ys, zs, step, dim, p1, p2;

    step = 0;
    dim = 3;

    path[step*dim+0] = x1;
    path[step*dim+1] = y1;
    path[step*dim+2] = z1;
    step++;

    dx = abs(x2-x1);
    dy = abs(y2-y1);
    dz = abs(z2-z1);

    xs = (x2>x1) ? 1 : -1;
    ys = (y2>y1) ? 1 : -1;
    zs = (z2>z1) ? 1 : -1;

    // Driving axis is X-axis
    if (dx >= dy && dx >= dz){
        p1 = 2*dy - dx;
        p2 = 2*dz - dx;

        while (x1 != x2){
            x1 = x1 + xs;
            if (p1 >= 0){
                y1 = y1 + ys;
                p1 = p1 - 2*dx;
            }
            if (p2 >= 0){
                z1 = z1 + zs;
                p2 = p2 - 2*dx;
            }
            p1 = p1 + 2*dy;
            p2 = p2 + 2*dz;
            path[step*dim+0] = x1;
            path[step*dim+1] = y1;
            path[step*dim+2] = z1;
            step++;
        }

    }
    // Driving axis is Y-axis
    else if (dy >= dx && dy >= dz){
        p1 = 2*dx - dy;
        p2 = 2*dz - dy;
        while (y1 != y2){
            y1 = y1 + ys;
            if (p1 >= 0){
                x1 = x1 + xs;
                p1 = p1 - 2*dy;
            }
            if (p2 >= 0){
                z1 = z1 + zs;
                p2 = p2 - 2*dy;
            }
            p1 = p1 + 2*dx;
            p2 = p2 + 2*dz;
            path[step*dim+0] = x1;
            path[step*dim+1] = y1;
            path[step*dim+2] = z1;
            step++;
        }

    }
    // Driving axis is Z-axis
    else{
        p1 = 2*dy - dz;
        p2 = 2*dx - dz;
        while (z1 != z2){
            z1 = z1 + zs;
            if (p1 >= 0){
                y1 = y1 + ys;
                p1 = p1 - 2*dz;
            }
            if (p2 >= 0){
                x1 = x1 + xs;
                p2 = p2 - 2*dz;
            }
            p1 = p1 + 2*dy;
            p2 = p2 + 2*dx;
            path[step*dim+0] = x1;
            path[step*dim+1] = y1;
            path[step*dim+2] = z1;
            step++;
        }
    }
    *steps = step*dim;

    return;
}


void L2_generate_paths(int dep, int row, int col, int imgdep, int imgrow, int imgcol, int step_in_image, int** paths, int* count_vox_in_paths, int *len_paths, int *num_paths){

    /* This function generate paths from [0,0,0] to the surface of the cuboid for all combinations between
    points [0,-row,-col] and [dep,row,col]. The rest of the not generated paths are considered symmetrical
    and therefore not necessary to evaluate. The paths can be shorter than the original size of the image,
    i.e. dep<=imgdep, row<=imgrow and col<=imgcol.

    dep - number of depths for the maximum path length
    row - number of rows for the maximum path length
    col - number of columns for the maximum path length
    imgdep - number of depths of the image
    imgrow - number of rows of the image
    imgcol - number of columns of the image
    paths - Bresenham's paths (2D dynamic array, each row represents one path, the length of each path can be variable)
    count_vox_in_paths - the number of repetition of each voxel in Bresenham's paths
    len_paths - lengths for each path (corresponds with the array paths in the same order)
    num_paths - total number of paths
    */

    /* initialization */
    int len_path, i, j, k, step, x, y, z, coord;
    int *path;

    int dim = 3;

    int diff_row[2] = {-row+1,row-1};  // puvodne do row+1
    int diff_col[2] = {-col+1,col-1};
    int diff_dep[2] = {0,dep-1};

    path = (int*) malloc(row*col*dep * sizeof(int));
     if (path == NULL) {
        printf("Memory not allocated.\n");
        exit(0);
    }

    step = 0;

    //######## path generation - axis 0 is constant ########

    for (i = 0; i<2; i++){
        for (j = -row+1; j<row; j+=step_in_image){
            for (k = -col+1; k<col; k+=step_in_image){
                Bresenham3D(0,0,0,diff_dep[i],j,k,path,&len_path);
                paths[step] = (int*) malloc(len_path * sizeof(int));
                memcpy(paths[step], path, len_path * sizeof *paths[step]);
                len_paths[step] = len_path;
                step++;
            }
        }
    }

    //######## path generation - axis 1 is constant ########

    for (i = 1; i<dep-1; i+=step_in_image){
        for (j = 0; j<2; j++){
            for (k = -col+1; k<col; k+=step_in_image){
                Bresenham3D(0,0,0,i,diff_row[j],k,path,&len_path);
                paths[step] = (int*) malloc(len_path * sizeof(int));
                memcpy(paths[step], path, len_path * sizeof *paths[step]);
                len_paths[step] = len_path;
                step++;
            }
        }
    }

    //######## path generation - axis 2 is constant ########
    for (i = 1; i<dep-1; i+=step_in_image){
        for (j = -row+2; j<row-1; j+=step_in_image){
            for (k = 0; k<2; k++){
                Bresenham3D(0,0,0,i,j,diff_col[k],path,&len_path);
                paths[step] = (int*) malloc(len_path * sizeof(int));
                memcpy(paths[step], path, len_path * sizeof *paths[step]);
                len_paths[step] = len_path;
                step++;
            }
        }
    }

    *num_paths = step;
    // how many times are all the paths repeated?
    for (i=0; i<step;i++){
        for (j=0; j<((int) len_paths[i]/dim); j++){
            x = paths[i][j*dim+0];
            y = paths[i][j*dim+1] + imgrow-1;
            z = paths[i][j*dim+2] + imgcol-1;
            coord = offset(x, y, z, 2*imgrow-1, 2*imgcol-1);
            count_vox_in_paths[coord] = count_vox_in_paths[coord] + imgdep*imgrow*imgcol;
        }
    }

    free(path);
    path = NULL;

return;
}

void L2_direct_computation(int *A, int dep, int row, int col, int phase, int** paths, int* count_vox_in_paths, int *len_paths, int num_paths, double* L2_mat_prob, int progress_flag, int start_dep, int stop_dep){

    /*
    A - 1D array with the medium saved along the rows
    dep - number of depths of the image
    row - number of rows of the image
    col - number of columns of the image
    phase - L2 is evaluated only for this phase
    paths - Bresenham's paths (2D dynamic array, each row represents one path, the length of each path can be variable)
    count_vox_in_paths - the number of repetition of each voxel in Bresenham's paths
    len_paths - lengths for each path (corresponds with the array paths in the same order)
    num_paths - total number of paths
    L2_mat_prob - Lineal path probability 3D matrix
    progress_flag - 1 is for saving progress of the L2 evaluation, slows the evaluation
    start_dep - the coordinate of the medium depth to start with, 0 for whole medium
    stop_dep  - the coordinate of the medium depth to end with, -1 for whole medium

    */

    int i, j, k, l, m, x, y, z, coord_A, coord_L2, shift;
    int dim = 3;
    int *L2_mat;
    double temp;
    int count = 0;
    double perc;
    FILE *fptr3;
    FILE *fid;

    if (progress_flag == 1) {

        fptr3 = fopen("TEMP-L2_progress.txt", "w");
            // exiting program
        if (fptr3 == NULL) {
            printf("Error!");
            exit(1);
        }

    }

    int count_voxels_half = dep * (2*row-1) * (2*col-1);
    L2_mat = (int*) calloc(count_voxels_half, sizeof(int));

    if (stop_dep == -1) {stop_dep = dep; }

    for (i = 0+start_dep; i<stop_dep; i++){
        for (j = row; j<2*row; j++){
            for (k = col; k<2*col; k++){

                for (l = 0; l<num_paths; l++){
                    if (A[(offset(i%dep,j%row,k%col,row,col))] == phase){
                        m = 0;
                        x = (paths[l][m*dim+0]);
                        y = (paths[l][m*dim+1]);
                        z = (paths[l][m*dim+2]);
                        coord_A = offset((i+x)%dep, (j+y)%row, (k+z)%col, row, col);

                        while (m<(len_paths[l]/dim) && A[coord_A] == phase){
                            coord_L2 = offset(x, y+row-1, z+col-1, 2*row-1, 2*col-1);
                            L2_mat[coord_L2]++;
                            m++;

                            x = (paths[l][m*dim+0]);
                            y = (paths[l][m*dim+1]);
                            z = (paths[l][m*dim+2]);
                            coord_A = offset((i+x)%dep, (j+y)%row, (k+z)%col, row, col);
                        }
                    }
                }

                if (progress_flag == 1) {
                    count++;
                    if (count % 100 == 0)
                    {
                        perc = (double)count/(double)(row*col*dep)*100;
                        fprintf(fptr3,"%d out of %d, %f%% done, dep: %d (%d - %d), row: %d, col: %d \n",count,dep*row*col,perc, i, start_dep, stop_dep, j, k);
                    }

                }
            }
        }
    }

    if (progress_flag == 1){
        fclose(fptr3);

        // save frequency matrix and the last scanned pixel
        FILE *fptr4;


        char name_file[32];
        sprintf(name_file, "TEMP-L2_freq_mat_%d-%d.dat", start_dep, stop_dep);
        printf("File name of the (partial) frequency matrix: %s \n", name_file);

        //fptr4 = fopen("TEMP-L2_freq_mat.txt", "w");
        fptr4 = fopen(name_file, "w");
        // exiting program
        if (fptr4 == NULL) {
            printf("Error!");
            exit(1);
        }

        for (l = 0; l<count_voxels_half; l++){
            fprintf(fptr4, "%d ",L2_mat[l]);
        }
        fprintf(fptr4, "\ndep: %d, row: %d, col: %d\n",i,j,k);
        fclose(fptr4);

        fid = fopen("TEMP-L2_possible_path_occurences.dat", "w");
        // exiting program
        if (fid == NULL) {
            printf("Error!");
            exit(1);
        }

    }

    shift = (dep-1) * (2*row-1) * (2*col-1);
    for (i = 0; i< ( dep * (2*row-1) * (2*col-1)) ; i++){
        if (count_vox_in_paths[i] != 0){
            temp = (double) (count_vox_in_paths[i] + 0.0);
            L2_mat_prob[i+shift] = L2_mat[i] / temp ;
            if (progress_flag == 1){
                fprintf(fid, "%d ",count_vox_in_paths[i]);
            }
        }
    }

    if (progress_flag == 1){
        fprintf(fid, "\n");
        fclose(fid);
    }

    for (i = 0; i< (dep-1); i++){
        for (j = 0; j<(2*row-1); j++){
            for (k = 0; k<(2*col-1); k++){
                L2_mat_prob[offset(i,j,k,2*row-1,2*col-1)] = L2_mat_prob[offset(2*dep-2-i,2*row-2-j,k,2*row-1,2*col-1)];
            }
        }
    }

    printf("Lineal path done\n");

    return;
}

void Lineal_path(int dep, int row, int col, int imgdep, int imgrow, int imgcol, int phase, int step_in_image, int *img_array, double *L2_mat_prob, int progress_flag, int start_dep, int stop_dep){

    int num_paths, num_paths_guess, count_voxels_half;
    int *len_paths, **paths, *count_vox_in_paths;
    float a, b, c, d;
    clock_t start, end;
    double cpu_time_used;

    if (step_in_image == 1){
        num_paths_guess  = (2*row-1)*(2*col-1)*2 + (2*row-1)*(dep-2)*2 + (2*col-3)*(dep-2)*2;
    }
    else {
        a = ceil((2*row-1) / (float) step_in_image);
        b = ceil((2*col-1) / (float) step_in_image);
        c = ceil((dep-2)   / (float) step_in_image);
        d = ceil((2*col-3) / (float) step_in_image);
        num_paths_guess = (int) a * (int) b * 2 + (int) a * (int) c * 2 + (int) d * (int) c * 2;
    }

    count_voxels_half = imgdep * (2*imgrow-1) * (2*imgcol-1);

    // 1-D arrays
    len_paths = (int*) malloc(num_paths_guess * sizeof(int));
    count_vox_in_paths = (int*) calloc(count_voxels_half, sizeof(int));

    // 2-D arrays
    paths = malloc(num_paths_guess * sizeof(*paths));

    if (len_paths == NULL || paths == NULL || count_vox_in_paths == NULL) {
        printf("Memory not allocated.\n");
        exit(0);
    }

    start = clock();
    L2_generate_paths(dep, row, col, imgdep, imgrow, imgcol, step_in_image, paths, count_vox_in_paths, len_paths, &num_paths);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("L2_generate_paths: %g\n",cpu_time_used);

   if (num_paths != num_paths_guess){
        printf("Warning: Evaluated number of paths is different from the real number of paths. \n");
        if (num_paths > num_paths_guess){
            printf("Warning: Allocated memory is smaller than needed memory for paths. Please, change step_in_image to a different number. \n");
        }
    }

    start = clock();
    L2_direct_computation(img_array, imgdep, imgrow, imgcol, phase, paths, count_vox_in_paths, len_paths, num_paths, L2_mat_prob, progress_flag, start_dep, stop_dep);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("L2_direct_computation: %g\n",cpu_time_used);

    /* free memory */
    free(len_paths);
    free(paths);
    free(count_vox_in_paths);
    len_paths = NULL;
    paths = NULL;
    count_vox_in_paths = NULL;

    printf("Done. Memory cleared. \n");
    //printf("Version from May 17, 15:24 \n");

    return ;
}
