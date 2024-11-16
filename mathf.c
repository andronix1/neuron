#include "mathf.h"

float randf(void) {
	return (float)rand() / RAND_MAX;
}

float sigmoidf(float x) {
	return 1.0 / (1.0 + expf(-x));
}

float sigmoidf_derivative(float x) {
	return expf(-x) / powf(1 + exp(-x), 2);
}
// f  = (1 + e ^ -x) ^ -1
// f' = - (1 + e ^ -x) ^ -2 * e ^ -x
