#pragma once

#include <math.h>

float sigmoidf(float x) {
	return 1.0 / (1.0 + expf(-x));
}
