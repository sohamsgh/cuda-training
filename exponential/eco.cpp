#include <cmath>
#include <iostream>
#include <stdbool.h>

#define EXP_A (1048576/M_LN2)
#define EXP_C 60801

inline 
double ecoexp(double y) {

	union {
		double d;
		struct {
#ifdef LITTLE_ENDIAN
		int j, i;
#else
		int i, j;
#endif
	} n;
	}
	_eco;

	_eco.n.i = (int) (EXP_A*(y)) + (1072693248 - EXP_C);
	_eco.n.j = 0;

	return _eco.d;
}

int main() {

	float delta = 0.01;
	bool flag = true;
	for (int i = -50; i<50; i+=10) {
		float x = ((float) i);
		double std_exp = exp(x);
		double fast_exp = ecoexp(x); 
		double err = fabs(std_exp - fast_exp);
		if (err > delta) {
			printf("*** FAILED ***\n");
			printf("x = %8.4f, exp(x) = %12.4f, ecoexp(x) = %12.4f, error = %e\n", x, std_exp, fast_exp, err);
			flag = false;
			break;
		}
	}
	if (flag == true) printf("Passed!\n");
	return 0;
}

