#include <iostream>
#include <fstream>
#include <vector>
#include "Kernel.cuh"

static void writePPM(const char* path, const std::vector<Pixel>& img, int W, int H) {
	std::ofstream f(path, std::ios::binary);
	f << "P6\n" << W << " " << H << "\n255\n";
	for (int i = 0; i < W * H; i++) {
		f.put((char)img[i].r);
		f.put((char)img[i].g);
		f.put((char)img[i].b);
	}
	f.close();
}

int main() {

	const int W = 1920;
	const int H = 1080;

	std::vector<Pixel> pixels(W * H);
	renderImage(pixels.data(), W, H);

	writePPM("render.ppm", pixels, W, H);
	std::cout << "Wrote render.ppm (" << W << "x" << H << ")\n";
	return 0;
}