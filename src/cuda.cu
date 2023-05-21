#include "cuda.cuh"
#include "helper.h"

#include <cstring>
#include <cmath>
#include <thrust/device_vector.h>
#include <thrust/scan.h>


void cpu_sort_pairs_cuda(float* keys_start, unsigned char* colours_start, int first, int last);


///
/// Algorithm storage
///
// Number of particles in d_particles
unsigned int cuda_particles_count;
__constant__ unsigned int D_PARTICLES_COUNT;
// Device pointer to a list of particles
Particle* d_particles;
// Device pointer to a histogram of the number of particles contributing to each pixel
unsigned int* d_pixel_contribs;
// Device pointer to an index of unique offsets for each pixels contributing colours
unsigned int* d_pixel_index;
// Device pointer to storage for each pixels contributing colours
unsigned char* d_pixel_contrib_colours;
// Device pointer to storage for each pixels contributing colours' depth
float* d_pixel_contrib_depth;
// The number of contributors d_pixel_contrib_colours and d_pixel_contrib_depth have been allocated for
unsigned int cuda_pixel_contrib_count;
// Host storage of the output image dimensions
int cuda_output_image_width;
int cuda_output_image_height;
// Device storage of the output image dimensions
__constant__ int D_OUTPUT_IMAGE_WIDTH;
__constant__ int D_OUTPUT_IMAGE_HEIGHT;
// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device// Pointer to device image data buffer, for storing the output image data, this must be passed to a kernel to be used on device
unsigned char* d_output_image_data;

//Host storage for when needed for functions
unsigned int* cpu_pixel_contribs;
unsigned int* cpu_pixel_index;
unsigned char* cpu_pixel_contrib_colours;
float* cpu_pixel_contrib_depth;

//Host storage needed for validation
unsigned int* h_pixel_contribs;
Particle* h_particles;
CImage cpu_output_image;

//Device details
cudaDeviceProp prop;
unsigned int maxThreads, maxThreadsPerBlock;

void checkCUDAError(const char*);

void cuda_begin(const Particle* init_particles, const unsigned int init_particles_count,
    const unsigned int out_image_width, const unsigned int out_image_height) {
    // These are basic CUDA memory allocations that match the CPU implementation
    // Depending on your optimisation, you may wish to rewrite these (and update cuda_end())

    // Allocate a opy of the initial particles, to be used during computation
    cuda_particles_count = init_particles_count;
    CUDA_CALL(cudaMemcpyToSymbol(D_PARTICLES_COUNT, &cuda_particles_count, sizeof(unsigned int)));

    CUDA_CALL(cudaMalloc(&d_particles, init_particles_count * sizeof(Particle)));
    CUDA_CALL(cudaMemcpy(d_particles, init_particles, init_particles_count * sizeof(Particle), cudaMemcpyHostToDevice));

    // Allocate a histogram to track how many particles contribute to each pixel
    CUDA_CALL(cudaMalloc(&d_pixel_contribs, out_image_width * out_image_height * sizeof(unsigned int)));
    // Allocate an index to track where data for each pixel's contributing colour starts/ends
    CUDA_CALL(cudaMalloc(&d_pixel_index, (out_image_width * out_image_height + 1) * sizeof(unsigned int)));
    cpu_pixel_index = (unsigned int*)malloc((out_image_width * out_image_height + 1) * sizeof(unsigned int));
    // Init a buffer to store colours contributing to each pixel into (allocated in stage 2)
    d_pixel_contrib_colours = 0;
    // Init a buffer to store depth of colours contributing to each pixel into (allocated in stage 2)
    d_pixel_contrib_depth = 0;
    // This tracks the number of contributes the two above buffers are allocated for, init 0
    cuda_pixel_contrib_count = 0;

    cpu_pixel_contrib_colours = 0;
    cpu_pixel_contrib_depth = 0;

    // Allocate output image
    cuda_output_image_width = (int)out_image_width;
    cuda_output_image_height = (int)out_image_height;
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_WIDTH, &cuda_output_image_width, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_HEIGHT, &cuda_output_image_height, sizeof(int)));
    const int CHANNELS = 3;  // RGB
    CUDA_CALL(cudaMalloc(&d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char)));

    cudaSetDevice(0);
    cudaGetDeviceProperties(&prop, 0);

    maxThreads = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;
    maxThreadsPerBlock = prop.maxThreadsPerBlock;


#ifdef VALIDATION
    cpu_output_image.width = (int)out_image_width;
    cpu_output_image.height = (int)out_image_height;
    cpu_output_image.channels = 3;  // RGB
    cpu_output_image.data = (unsigned char*)malloc(cpu_output_image.width * cpu_output_image.height * cpu_output_image.channels * sizeof(unsigned char));
#endif

}

__global__ void kernal_stage1(const Particle* __restrict__ d_particles, unsigned int* d_pixel_contribs) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute bounding box [inclusive-inclusive]
    int x_min = (int)roundf(d_particles[i].location[0] - d_particles[i].radius);
    int y_min = (int)roundf(d_particles[i].location[1] - d_particles[i].radius);
    int x_max = (int)roundf(d_particles[i].location[0] + d_particles[i].radius);
    int y_max = (int)roundf(d_particles[i].location[1] + d_particles[i].radius);
    // Clamp bounding box to image bounds
    x_min = x_min < 0 ? 0 : x_min;
    y_min = y_min < 0 ? 0 : y_min;
    x_max = x_max >= D_OUTPUT_IMAGE_WIDTH ? D_OUTPUT_IMAGE_HEIGHT - 1 : x_max;
    y_max = y_max >= D_OUTPUT_IMAGE_WIDTH ? D_OUTPUT_IMAGE_HEIGHT - 1 : y_max;
    // For each pixel in the bounding box, check that it falls within the radius
    for (int x = x_min; x <= x_max; ++x) {
        for (int y = y_min; y <= y_max; ++y) {
            const float x_ab = (float)x + 0.5f - d_particles[i].location[0];
            const float y_ab = (float)y + 0.5f - d_particles[i].location[1];
            const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
            if (pixel_distance <= d_particles[i].radius) {
                const unsigned int pixel_offset = y * D_OUTPUT_IMAGE_WIDTH + x;
                atomicAdd(&d_pixel_contribs[pixel_offset], 1);
            }
        }
    }
}
void cuda_stage1() {
    CUDA_CALL(cudaMemset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int)));
    unsigned int blocksPerGrid = cuda_particles_count / maxThreadsPerBlock + 1;
    unsigned int threadsPerBlock = maxThreadsPerBlock;
    kernal_stage1 << <blocksPerGrid, threadsPerBlock >> > (d_particles, d_pixel_contribs);


#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    h_pixel_contribs = (unsigned int*)malloc(cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int));
    h_particles = (Particle*)malloc(cuda_particles_count * sizeof(Particle));
    cudaMemcpy(h_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_particles, d_particles, cuda_particles_count * sizeof(Particle), cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy");
    validate_pixel_contribs(h_particles, cuda_particles_count, h_pixel_contribs, cuda_output_image_width, cuda_output_image_height);
#endif
}

__global__ void kernal_stage2(const Particle* __restrict__ d_particles, unsigned int* d_pixel_contribs, unsigned int* d_pixel_index, unsigned char* d_pixel_contrib_colours, float* d_pixel_contrib_depth) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D_PARTICLES_COUNT) {
        return;
    }
    const Particle p = d_particles[i];

    int x_min = (int)roundf(p.location[0] - p.radius);
    int y_min = (int)roundf(p.location[1] - p.radius);
    int x_max = (int)roundf(p.location[0] + p.radius);
    int y_max = (int)roundf(p.location[1] + p.radius);

    // Clamp bounding box to image bounds
    x_min = x_min < 0 ? 0 : x_min;
    y_min = y_min < 0 ? 0 : y_min;
    x_max = x_max >= D_OUTPUT_IMAGE_WIDTH ? D_OUTPUT_IMAGE_HEIGHT - 1 : x_max;
    y_max = y_max >= D_OUTPUT_IMAGE_WIDTH ? D_OUTPUT_IMAGE_HEIGHT - 1 : y_max;

    // For each pixel in the bounding box, check that it falls within the radius
    for (int x = x_min; x <= x_max; ++x) {
        for (int y = y_min; y <= y_max; ++y) {
            const float x_ab = (float)x + 0.5f - d_particles[i].location[0];
            const float y_ab = (float)y + 0.5f - d_particles[i].location[1];
            const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);

            if (pixel_distance <= p.radius) {
                const unsigned int pixel_offset = y * D_OUTPUT_IMAGE_WIDTH + x;
                const unsigned int index_offset = d_pixel_index[pixel_offset] + atomicAdd(&d_pixel_contribs[pixel_offset], 1);
                // Copy data to d_pixel_contrib buffers
                d_pixel_contrib_colours[4 * index_offset + 0] = p.color[0];
                d_pixel_contrib_colours[4 * index_offset + 1] = p.color[1];
                d_pixel_contrib_colours[4 * index_offset + 2] = p.color[2];
                d_pixel_contrib_colours[4 * index_offset + 3] = p.color[3];

                d_pixel_contrib_depth[index_offset] = p.location[2];
            }
        }
    }
}
void cuda_stage2() {
    thrust::device_ptr<unsigned int> d_pixel_index_thrust(d_pixel_index);
    thrust::device_ptr<unsigned int> d_pixel_contribs_thrust(d_pixel_contribs);
    thrust::exclusive_scan(d_pixel_contribs_thrust, d_pixel_contribs_thrust + cuda_output_image_width * cuda_output_image_height + 1, d_pixel_index_thrust);


    unsigned int TOTAL_CONTRIBS;
    cudaMemcpy(&TOTAL_CONTRIBS, d_pixel_index + cuda_output_image_width * cuda_output_image_height, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    if (TOTAL_CONTRIBS > cuda_pixel_contrib_count) {
        cuda_pixel_contrib_count = TOTAL_CONTRIBS;
    }

    if (cpu_pixel_contrib_colours) free(cpu_pixel_contrib_colours);
    if (cpu_pixel_contrib_depth) free(cpu_pixel_contrib_depth);
    cpu_pixel_contrib_colours = (unsigned char*)malloc(cuda_pixel_contrib_count * 4 * sizeof(unsigned char));
    cpu_pixel_contrib_depth = (float*)malloc(cuda_pixel_contrib_count * sizeof(float));


    CUDA_CALL(cudaMemset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int)));
    //memcpy colours and depth to gpu
    cudaMalloc(&d_pixel_contrib_colours, cuda_pixel_contrib_count * 4 * sizeof(unsigned char));
    cudaMalloc(&d_pixel_contrib_depth, cuda_pixel_contrib_count * sizeof(float));
    cudaMemcpy(d_pixel_contrib_colours, cpu_pixel_contrib_colours, cuda_pixel_contrib_count * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pixel_contrib_depth, cpu_pixel_contrib_depth, cuda_pixel_contrib_count * sizeof(float), cudaMemcpyHostToDevice);

    unsigned int blocksPerGrid = cuda_particles_count / maxThreadsPerBlock + 1;
    unsigned int threadsPerBlock = maxThreadsPerBlock;
    kernal_stage2 << <blocksPerGrid, threadsPerBlock >> > (d_particles, d_pixel_contribs, d_pixel_index, d_pixel_contrib_colours, d_pixel_contrib_depth);

    // Copy data back to host
    cudaMemcpy(cpu_pixel_contrib_colours, d_pixel_contrib_colours, cuda_pixel_contrib_count * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_pixel_contrib_depth, d_pixel_contrib_depth, cuda_pixel_contrib_count * sizeof(float), cudaMemcpyDeviceToHost);
    //copy cpu index back to host
    cudaMemcpy(cpu_pixel_index, d_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);


    // Pair sort the colours contributing to each pixel based on ascending depth
    for (int i = 0; i < cuda_output_image_width * cuda_output_image_height; ++i) {
        // Pair sort the colours which contribute to a single pigment
        cpu_sort_pairs_cuda(
            cpu_pixel_contrib_depth,
            cpu_pixel_contrib_colours,
            cpu_pixel_index[i],
            cpu_pixel_index[i + 1] - 1
        );
    }

    //memcpy back

    cudaMemcpy(d_pixel_contrib_colours, cpu_pixel_contrib_colours, cuda_pixel_contrib_count * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pixel_contrib_depth, cpu_pixel_contrib_depth, cuda_pixel_contrib_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pixel_index, cpu_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice);




#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // Note: Only validate_equalised_histogram() MUST be uncommented, the others are optional
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    cudaMemcpy(h_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_particles, d_particles, cuda_particles_count * sizeof(Particle), cudaMemcpyDeviceToHost);

    checkCUDAError("CUDA memcpy");
    validate_pixel_index(h_pixel_contribs, cpu_pixel_index, cuda_output_image_width, cuda_output_image_height);
    validate_sorted_pairs(h_particles, cuda_particles_count, cpu_pixel_index, cuda_output_image_width, cuda_output_image_height, cpu_pixel_contrib_colours, cpu_pixel_contrib_depth);
#endif    
}
__global__ void kernal_stage3(unsigned char* d_output_image_data, unsigned int* d_pixel_index, unsigned char* d_pixel_contrib_colours) {


    int i = blockIdx.x * blockDim.x + threadIdx.x;


    unsigned int first = __ldg(&d_pixel_index[i]);
    unsigned int last = __ldg(&d_pixel_index[i + 1]);
    if (i < D_OUTPUT_IMAGE_WIDTH * D_OUTPUT_IMAGE_HEIGHT) {
        for (int j = first; j < last; ++j) {
            const float opacity = (float)d_pixel_contrib_colours[j * 4 + 3] / (float)255;
            d_output_image_data[(i * 3) + 0] = (unsigned char)((float)d_pixel_contrib_colours[j * 4 + 0] * opacity + (float)d_output_image_data[(i * 3) + 0] * (1 - opacity));
            d_output_image_data[(i * 3) + 1] = (unsigned char)((float)d_pixel_contrib_colours[j * 4 + 1] * opacity + (float)d_output_image_data[(i * 3) + 1] * (1 - opacity));
            d_output_image_data[(i * 3) + 2] = (unsigned char)((float)d_pixel_contrib_colours[j * 4 + 2] * opacity + (float)d_output_image_data[(i * 3) + 2] * (1 - opacity));
        }
    }
}
void cuda_stage3() {
    cudaMemset(d_output_image_data, 255, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char));
    dim3 blocksPerGrid((unsigned int)ceil(cuda_output_image_width * cuda_output_image_height / (double)128), 1, 1);
    dim3 threadsPerBlock(128, 1, 1);
    kernal_stage3 << <blocksPerGrid, threadsPerBlock >> > (d_output_image_data, d_pixel_index, d_pixel_contrib_colours);



#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    // You will need to copy the data back to host before passing to these functions
    // (Ensure that data copy is carried out within the ifdef VALIDATION so that it doesn't affect your benchmark results!)
    cudaMemcpy(cpu_pixel_contrib_colours, d_pixel_contrib_colours, cuda_pixel_contrib_count * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(cpu_pixel_index, d_pixel_index, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    //cpy to cpu output image
    cudaMemcpy(cpu_output_image.data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    checkCUDAError("CUDA memcpy");
    validate_blend(cpu_pixel_index, cpu_pixel_contrib_colours, &cpu_output_image);
#endif    
}
void cuda_end(CImage* output_image) {
    // This function matches the provided cuda_begin(), you may change it if desired

    // Store return value
    const int CHANNELS = 3;
    output_image->width = cuda_output_image_width;
    output_image->height = cuda_output_image_height;
    output_image->channels = CHANNELS;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    // Release allocations
    CUDA_CALL(cudaFree(d_pixel_contrib_depth));
    CUDA_CALL(cudaFree(d_pixel_contrib_colours));
    CUDA_CALL(cudaFree(d_output_image_data));
    CUDA_CALL(cudaFree(d_pixel_index));
    CUDA_CALL(cudaFree(d_pixel_contribs));
    CUDA_CALL(cudaFree(d_particles));
    free(cpu_pixel_contribs);
    free(cpu_pixel_index);
    free(cpu_pixel_contrib_colours);
    free(cpu_pixel_contrib_depth);
    // Return ptrs to nullptr
    d_pixel_contrib_depth = 0;
    d_pixel_contrib_colours = 0;
    d_output_image_data = 0;
    d_pixel_index = 0;
    d_pixel_contribs = 0;
    d_particles = 0;
    cpu_pixel_contrib_colours = 0;
    cpu_pixel_contrib_depth = 0;
    cpu_pixel_contribs = 0;
    cpu_pixel_index = 0;
}

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void cpu_sort_pairs_cuda(float* keys_start, unsigned char* colours_start, const int first, const int last) {
    // Based on https://www.tutorialspoint.com/explain-the-quick-sort-technique-in-c-language
    int i, j, pivot;
    float depth_t;
    unsigned char color_t[4];
    if (first < last) {
        pivot = first;
        i = first;
        j = last;
        while (i < j) {
            while (keys_start[i] <= keys_start[pivot] && i < last)
                i++;
            while (keys_start[j] > keys_start[pivot])
                j--;
            if (i < j) {
                // Swap key
                depth_t = keys_start[i];
                keys_start[i] = keys_start[j];
                keys_start[j] = depth_t;
                // Swap color
                memcpy(color_t, colours_start + (4 * i), 4 * sizeof(unsigned char));
                memcpy(colours_start + (4 * i), colours_start + (4 * j), 4 * sizeof(unsigned char));
                memcpy(colours_start + (4 * j), color_t, 4 * sizeof(unsigned char));
            }
        }
        // Swap key
        depth_t = keys_start[pivot];
        keys_start[pivot] = keys_start[j];
        keys_start[j] = depth_t;
        // Swap color
        memcpy(color_t, colours_start + (4 * pivot), 4 * sizeof(unsigned char));
        memcpy(colours_start + (4 * pivot), colours_start + (4 * j), 4 * sizeof(unsigned char));
        memcpy(colours_start + (4 * j), color_t, 4 * sizeof(unsigned char));
        // Recurse
        cpu_sort_pairs_cuda(keys_start, colours_start, first, j - 1);
        cpu_sort_pairs_cuda(keys_start, colours_start, j + 1, last);
    }
}