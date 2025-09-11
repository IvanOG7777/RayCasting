#pragma once
#include <cuda_runtime.h>
#include <iostream>

// creates pixel struct
// each value is uint8_t that will hold value from 0-255
// used to control coloring in graphics apis like openGL,
// where the higher the number the more intense that color will be
struct Pixel {
	uint8_t r;
	uint8_t g;
	uint8_t b;
	uint8_t a;
};

void renderImage(Pixel* hostPixels, int width, int height);