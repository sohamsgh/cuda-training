#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include "utils.h"

#define EPSILON 1e-06

void compare_matrices(float *res, float*ref, int M, int N) {
	bool passed = true;
	for (int row = 0; row < M; row++) {
		for (int col = 0; col < N; col++){
			int i = row * N + col;
			if (fabs(res[i] -ref[i]) > EPSILON) {
				printf("%s \n %s \n", "***FAILED ***", "result matrix validation failed at");
            			printf("(row, col): %d, %d, Reference: %f, result: %f\n", row, col,ref[i], res[i]);
            			passed = false;
            			break;
			}
		}
		if (!passed) {
			break;
		}
	}
        if (passed == true) {
            printf("Matrix validated\n");
        }
}

void softmax_cpu(float *input, float *result, int M, int N) {

	for (int row = 0; row < M; row++) { // loop over rows
		float row_max = -1*INFINITY;
		float row_norm = 0.0f;
		// first get the max for each row
		for (int col = 0; col < N; col++) {
			int i = row * N + col;
			row_max = fmax(row_max, input[i]);
		}
		// now get the norm
		for (int col = 0; col < N; col++) {
			int i = row * N + col;
			row_norm += expf(input[i] - row_max);
		}
		// finally the softmax
		for (int col = 0; col < N; col++) {
			int i = row * N + col;
			result[i] = expf(input[i] - row_max) / row_norm;
		}
	}
}


			
