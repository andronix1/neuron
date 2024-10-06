#include "rand.h"

float randf(void) {
	return (float)rand() / RAND_MAX;
}
