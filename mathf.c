#include "mathf.h"

float randf(void) {
	return (float)rand() / RAND_MAX;
}

float sigmoidf(float x) {
	return 1.0 / (1.0 + expf(-x));
}
