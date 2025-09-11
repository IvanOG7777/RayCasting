#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cmath>
#include "Kernel.cuh"

// struct used to group up x y z position
// all are of type float
struct float3x {
	float x;
	float y;
	float z;
};

// function used to create a float3x type
// instead of writing float3x v
// v.x = 1.0f;
// v.y = 2.0f;
// v.z = 3.0f;

// we just pass in the numbers and return the whole object
__host__ __device__ inline float3x makef3(float x, float y, float z) {
	return { x, y, z };
}

// function used to add two values of float3x
__host__ __device__ inline float3x operator+(const float3x& a, const float3x& b) { // pass by reference a and b
	return makef3(a.x + b.x, a.y + b.y, a.z + b.z); // return the sum of all x,y,z values in a and b
}

// function used to subtract two values of float3x
__host__ __device__ inline float3x operator-(const float3x& a, const float3x& b) {  // pass by reference a and b
	return makef3(a.x - b.x, a.y - b.y, a.z - b.z); // return the sum of all x,y,z values in a and b
}

// function used to mutiple a vector of float3x by a scalar
__host__ __device__ inline float3x operator* (const float3x& a, float scalar) { // pass in the vector of floats and a scalar
	return makef3(a.x * scalar, a.y * scalar, a.z * scalar); // return a single new vector of flaot3x but scaled up
}

__host__ __device__ inline float3x operator*(float scalar, const float3x& a) { // pass in the scalar of floats and the vector of floats
	return a * scalar;
}

// function used to computer the dot product of two float3x vectors
__host__ __device__ inline float dotProduct(const float3x& a, const float3x& b) { // pass in float3x a and b
	return (a.x * b.x) + (a.y * b.y) + (a.z * b.z); // return single float value after computation
}

// function used to calculate the magniture length of a vector  float3x
__host__ __device__ inline float magnitudeLength(const float3x& a) { // pass in a single float3x value
	return sqrtf(dotProduct(a, a)); // compuer dot product of a then sqrt it
	// ex v = make3f (1.0, 2.0, 3.0);
	// dotProduct (v, v);
	// newV = (1.0, 4.0, 9.0);
	// newVSum = 14
	// sqrft(newVSum) = 3.7417
}


__host__ __device__ float3x normalize(const float3x& a) {
	float length = magnitudeLength(a);

	// if length is larger than 0 reutrn a * (1.0f / length) else return a
	return (length > 0) ? a * (1.0f / length) : a;
}

__host__ __device__ inline float3x crossProduct(const float3x& a, const float3x& b) {
	return makef3((a.y * b.z) - (a.z * b.y), (a.z * b.x) - (a.x * b.z), (a.x * b.y) - (a.y * b.x));
}

__host__ __device__ inline float3x clamp01(const float3x &c) {
	return makef3(
		// used to clamp between [0...1]
		//if c.n is less than 0.0 clamp that value to 0.0;
		//if c.n is greater than 1.0 clamp that value to 1.0;
		fminf(fmaxf(c.x, 0.f), 1.f),
		fminf(fmaxf(c.y, 0.f), 1.f),
		fminf(fmaxf(c.z, 0.f), 1.f));
}

struct Ray {
	float3x origin;
	float3x direction;
};


// declaration to create a sphere object
struct Sphere {
	float3x center; // position of sphere in 3D space
	float radius; // radius of shpere
	float3x color; // RGB color
};

struct Plane {
	float3x normal; // orientation of the plane
	float distance; // distance form origin
	float3x colorA; // one color (checker pattern)
	float3x colorB; // alternate color
	float cell = 1.0f; // size of checker cells
};

struct Camera {
	float3x eye; // camera position
	float3x u; // right vector (x-axis of camera space)
	float3x v; // up vector (y-axis of camera space)
	float3x w; // backwards vector (z-axis of camera space)

	float tanHalFov; // tangent of half the field of view
	float aspectRatio; // width / height
}; 

__host__ __device__ inline Ray makePrimaryRay(const Camera& camera, int x, int y, int W, int H) {
	float px = ((x + 0.5f) / (float)W * 2.f - 1.f) * camera.aspectRatio * camera.tanHalFov;

	float py = (1.f - (y + 0.5f) / (float)H * 2.f) * camera.tanHalFov;

	float3x direction = normalize(camera.u * px + camera.v * py - camera.w);
	return {camera.eye, direction}; // return the camera.eye as the origin and direction as direction
}