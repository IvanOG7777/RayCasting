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

// function to normalize a vector between -1 <-> 0 <-> 1 in both x,y,z axis
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
	float radius; // radius of sphere
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

// function used to calculate the origin of where the ray is being sent from realtive to the cameras eye
__host__ __device__ inline Ray makePrimaryRay(const Camera& camera, int x, int y, int W, int H) {
	float px = ((x + 0.5f) / (float)W * 2.f - 1.f) * camera.aspectRatio * camera.tanHalFov;

	float py = (1.f - (y + 0.5f) / (float)H * 2.f) * camera.tanHalFov;

	float3x direction = normalize(camera.u * px + camera.v * py - camera.w);
	return {camera.eye, direction}; // return the camera.eye as the origin and direction as direction
}

// function used to detect if ray has hit a sphere
// pass in the a Sphere and Ray object, minimum distance ray has to travel (tMin), maximum distance t has to travel (maxT), output distance where ray hit (outT), and a unit vector pointing outward from the surface (outNormal)
// outT adn outNormal will usually be uninitalized since the function will handle its new values then return them by reference
__device__ bool hitSphere(const Sphere& sphere, const Ray& ray, float tMin, float tMax, float& outT, float3x& outNormal) {

	// calculates the distance from the rays origin to the spheres center coordinates
	float3x originToCenter = ray.origin - sphere.center;

	// Quadratic coefficients to calculate the intersection of ray
	float a = dotProduct(ray.direction, ray.direction); // squared length of the rays direction
	float b = 2.f * dotProduct(originToCenter, ray.direction); // How much the rays origin directionally aligns with the vector from the spheres center
	float c = dotProduct(originToCenter, originToCenter) - sphere.radius * sphere.radius; // squared distance from ray origin minus the spheres radius squared

	float discriminant = b * b - 4 * a * c; // uses quadratic formula to calculate discriminant gives us two results +-disc

	if (discriminant < 0) return false; // if disc is less than 0 return we didnt hit enything

	float sqrtDisc = sqrtf(discriminant); // sqrt the disc

	float tNear = (-b - sqrtDisc) / (2 * a); // get closest hit on sphere

	float tFar = (-b + sqrtDisc) / (2 * a); // get the farthest hit on sphere

	float hitT = tNear; // hitT is the closest hit on sphere

	// bounds checker
	if (hitT < tMin || hitT > tMax) { // if hitT is smaller than minimum distance or farther than maximum distance essentially checks if near is valid
		hitT = tFar; // if near is not valid hitT is set to the farthest value ray is allowed to travel
		if (hitT < tMin || hitT > tMax) return false; // test again and if this fails return false (no hit)
	}

	outT = hitT; // if bounds cheker passes outT is set to hiT distance (passed by referecne)

	float3x hitPoint = ray.origin + ray.direction * outT; // calculate the ray.directions x,y,z values by multiplying by scalar (outT) then adding to ray.orgin
	// gives up the x,y,z coordnites of where ray has hit

	outNormal = normalize(hitPoint - sphere.center); // normalize it between -1 <-> 0 <-> 1 on x,y,z axis
	return true;
}

// function used to calculate the if a ray its the plane wihin the min and max T
// passes in a reference of a Plane and Ray object, min and max T distance, a reference of a normal vector, colorOut of type float3x and a float cell
__device__ bool hitPlane(const Plane& plane, const Ray& ray,float tMin, float tMax, float& outT, float3x& normal, float3x& colorOut, float cell) {

	float denominator = dotProduct(plane.normal, ray.direction); // calculates the denominator using dot product of planes normal vector and rays direction
	// if denominator is close to 0, the ray is essentially parallel to the plane (no hit or infinite hits if lying in the plane)

	//fabsf = floating-point absolute value of a float
	if (fabsf(denominator) < 1e-5f) return false; // parallel checker. If its less than 1e-5f we return since its parallel

	float hitT = -(dotProduct(plane.normal, ray.origin) + plane.distance) / denominator; // compute the dot product of (plane.normal and ray.origin) + plane.distance all divided by denominator to get a hit

	if (hitT < tMin || hitT > tMax) return false; // bound checker to see if hitT is within bounds

	outT = hitT; // outT set to hitT

	normal = plane.normal;

	float3x hit = ray.origin + ray.direction * outT;
	int cx = (int)floorf(hit.x / cell);
	int cz = (int)floorf(hit.z / cell);
	bool isA = ((cx + cz) & 1) == 0;
	colorOut = isA ? plane.colorA : plane.colorB;
	return true;
}

__device__ bool shadowAnyHit(const Ray& shadowRay, const Sphere* spheres, int sphereCount, float lightDist) {
	
	float tDummy;
	float3x normalDummy;

	for (int i = 0; i < sphereCount; i++) {
		if (hitSphere(spheres[i], shadowRay, 1e-3f, lightDist - 1e-3f, tDummy, normalDummy))
			return true;
	}
	return false;
}

__device__ float3x skyColor(const float3x& d) {
	float t = 0.5f * (d.y + 1.f);
	return (1.f - t) * makef3(1.f, 1.f, 1.f) + t * makef3(0.5f, 0.7f, 1.0f);
}

struct Scene {
	Sphere* spheres;
	int sphereCount;
	Plane plane;
	float3x lightPos;
};

__device__ float3x shade(const Scene& scene, const Ray& ray) {
	
	float tHit = 1e30f;
	float3x Normal;
	float3x baseColor;
	bool hit = false;

	for (int i = 0; i < scene.sphereCount; i++) {
		float t;
		float3x normal;

		if (hitSphere(scene.spheres[i], ray, 1e-3f, tHit, t, normal)) {
			tHit = t;
			Normal = normal;
			baseColor = scene.spheres[i].color;
			hit = true;
		}
	}

	float t;
	float3x normal;
	float3x planeCol;

	if (hitPlane(scene.plane, ray, 1e-3f, tHit, t, normal, planeCol, scene.plane.cell)) {
		tHit = t;
		Normal = normal;
		baseColor = planeCol;
		hit = true;
	}

	if (!hit) return skyColor(ray.direction);

	float3x P = ray.origin + ray.direction * tHit;
	float3x L = scene.lightPos - P;
	float distL = magnitudeLength(L);
	L = L * (1.0f / (distL + 1e-6f));

	Ray sray{ P + Normal * 1e-3f, L };
	bool inShadow = shadowAnyHit(sray, scene.spheres, scene.sphereCount, distL);

	float ambient = 0.1f;
	float diff = fmaxf(dotProduct(Normal, L), 0.f);
	float kd = inShadow ? 0.f : 0.9f;

	float3x c = baseColor * (ambient + kd * diff);

	float att = 1.0f / (1.0f + 0.05f * distL + 0.005f * distL * distL);
	return clamp01(c * att);
}

__global__ void renderKernel(Pixel* out, int W, int H, Sphere* spheres, int sphereCount, Plane plane, Camera camera, float3x lightpos) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= W || y >= H) return;

	Ray pr = makePrimaryRay(camera, x, y, W, H);
	Scene scence{ spheres, sphereCount, plane, lightpos };
	float3x col = shade(scence, pr);

	int index = y * W + x;

	out[index].r = (uint8_t)(255.f * col.x);
	out[index].g = (uint8_t)(255.f * col.y);
	out[index].b = (uint8_t)(255.f * col.z);
	out[index].a = 255;
}

static void check(cudaError_t e) {
	if (e != cudaSuccess) {
		fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
		abort();
	}
}

void renderImage(Pixel* hostPixels, int width, int height) {
	
	float fovDegree = 60.0f;
	float tangentHalf = tanf(0.5f * fovDegree * (3.14159265f / 180.f));
	float aspectRatio = (float)width / (float)height;

	float3x eye = makef3(0, 0.5f, 3.2f);
	float3x target = makef3(0, 0.0f, 0.0f);
	float3x up = makef3(0, 1, 0);

	float3x w = normalize(eye - target);
	float3x u = normalize(crossProduct(up, w));
	float3x v = crossProduct(w, u);

	Camera camera{ eye, u, v, w, tangentHalf, aspectRatio };


	Sphere hostSpheres[] = {
		{ makef3(-0.8f, 0.0f, 0.0f), 0.5f, makef3(0.95f, 0.2f, 0.2f) },
		{ makef3(0.8f, 0.0f, 0.2f), 0.5f, makef3(0.2f, 0.9f, 0.3f) },
		{ makef3(0.0f, 0.6f,-0.6f), 0.4f, makef3(0.2f, 0.4f, 1.0f) }
	};

	int sphereCount = int(sizeof(hostSpheres) / sizeof(Sphere));

	Plane plane;
	plane.normal = makef3(0, 1, 0);
	plane.distance = 1.0f; // y = -1 plane
	plane.colorA = makef3(0.9f, 0.9f, 0.9f);
	plane.colorB = makef3(0.2f, 0.2f, 0.2f);
	plane.cell = 0.5f;

	float3x lightPos = makef3(5, 5, 3);

	Pixel* dOut = nullptr;
	Sphere* deviceSpheres = nullptr;

	size_t outBytes = (size_t)width * height * sizeof(Pixel);

	check(cudaMalloc(&dOut, outBytes));
	check(cudaMalloc(&deviceSpheres, sizeof(hostSpheres)));

	check(cudaMemcpy(deviceSpheres, hostSpheres, sizeof(hostSpheres), cudaMemcpyHostToDevice));

	dim3 block (16, 16);
	dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

	renderKernel <<<grid, block>>> (dOut, width, height, deviceSpheres, sphereCount, plane, camera, lightPos);
	check(cudaPeekAtLastError());
	check(cudaDeviceSynchronize());

	check(cudaMemcpy(hostPixels, dOut, outBytes, cudaMemcpyDeviceToHost));

	cudaFree(dOut);
	cudaFree(deviceSpheres);

}