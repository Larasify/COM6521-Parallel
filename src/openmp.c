#include "openmp.h"
#include "helper.h"

#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>



unsigned int omp_particles_count;
Particle* omp_particles;
unsigned int* omp_pixel_contribs;
unsigned int* omp_pixel_index;
unsigned char* omp_pixel_contrib_colours;
float* omp_pixel_contrib_depth;
unsigned int omp_pixel_contrib_count;
CImage omp_output_image;


void cpu_sort_pairs_openmp(float* keys_start, unsigned char* colours_start, int first, int last);

void openmp_begin(const Particle* init_particles, const unsigned int init_particles_count,
    const unsigned int out_image_width, const unsigned int out_image_height) {
    omp_particles_count = init_particles_count;
    omp_particles = malloc(init_particles_count * sizeof(Particle));
    memcpy(omp_particles, init_particles, init_particles_count * sizeof(Particle));


    // Allocate a histogram to track how many particles contribute to each pixel
    omp_pixel_contribs = (unsigned int*)malloc(out_image_width * out_image_height * sizeof(unsigned int));
    // Allocate an index to track where data for each pixel's contributing colour starts/ends
    omp_pixel_index = (unsigned int*)malloc((out_image_width * out_image_height + 1) * sizeof(unsigned int));
    // Init a buffer to store colours contributing to each pixel into (allocated in stage 2)
    omp_pixel_contrib_colours = 0;
    // Init a buffer to store depth of colours contributing to each pixel into (allocated in stage 2)
    omp_pixel_contrib_depth = 0;
    // This tracks the number of contributes the two above buffers are allocated for, init 0
    omp_pixel_contrib_count = 0;

    // Allocate output image
    omp_output_image.width = (int)out_image_width;
    omp_output_image.height = (int)out_image_height;
    omp_output_image.channels = 3;  // RGB
    omp_output_image.data = (unsigned char*)malloc(omp_output_image.width * omp_output_image.height * omp_output_image.channels * sizeof(unsigned char));
}
void openmp_stage1() {
    memset(omp_pixel_contribs, 0, omp_output_image.width * omp_output_image.height * sizeof(unsigned int));
    // Update each particle & calculate how many particles contribute to each image
    int i, x_min, y_min, x_max, y_max;
#pragma omp parallel for private(i, x_min, y_min, x_max, y_max) schedule(dynamic,8)
    for (i = 0; i < omp_particles_count; ++i) {
        // Compute bounding box [inclusive-inclusive]
        x_min = (int)roundf(omp_particles[i].location[0] - omp_particles[i].radius);
        y_min = (int)roundf(omp_particles[i].location[1] - omp_particles[i].radius);
        x_max = (int)roundf(omp_particles[i].location[0] + omp_particles[i].radius);
        y_max = (int)roundf(omp_particles[i].location[1] + omp_particles[i].radius);
        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= omp_output_image.width ? omp_output_image.width - 1 : x_max;
        y_max = y_max >= omp_output_image.height ? omp_output_image.height - 1 : y_max;
        // For each pixel in the bounding box, check that it falls within the radius
        int x, y;
        float x_ab, y_ab, pixel_distance;
        for (x = x_min; x <= x_max; ++x) {
            for (y = y_min; y <= y_max; ++y) {
                x_ab = (float)x + 0.5f - omp_particles[i].location[0];
                y_ab = (float)y + 0.5f - omp_particles[i].location[1];
                pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= omp_particles[i].radius) {
                    const unsigned int pixel_offset = y * omp_output_image.width + x;
#pragma omp atomic
                    ++omp_pixel_contribs[pixel_offset];
                }
            }
        }
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_pixel_contribs(omp_particles, omp_particles_count, omp_pixel_contribs, omp_output_image.width, omp_output_image.height);
#endif
}
void openmp_stage2() {
    omp_pixel_index[0] = 0;
    for (int i = 0; i < omp_output_image.width * omp_output_image.height; ++i) {
        omp_pixel_index[i + 1] = omp_pixel_index[i] + omp_pixel_contribs[i];
    }
    // Recover the total from the index
    const unsigned int TOTAL_CONTRIBS = omp_pixel_index[omp_output_image.width * omp_output_image.height];
    if (TOTAL_CONTRIBS > omp_pixel_contrib_count) {
        // (Re)Allocate colour storage
        if (omp_pixel_contrib_colours) free(omp_pixel_contrib_colours);
        if (omp_pixel_contrib_depth) free(omp_pixel_contrib_depth);
        omp_pixel_contrib_colours = (unsigned char*)malloc(TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
        omp_pixel_contrib_depth = (float*)malloc(TOTAL_CONTRIBS * sizeof(float));
        omp_pixel_contrib_count = TOTAL_CONTRIBS;
    }

    // Reset the pixel contributions histogram
    memset(omp_pixel_contribs, 0, omp_output_image.width * omp_output_image.height * sizeof(unsigned int));
    // Store colours according to index
    // For each particle, store a copy of the colour/depth in cpu_pixel_contribs for each contributed pixel
    omp_set_nested(1);
    int i = 0;
#pragma omp parallel for schedule(dynamic,8)
    for (i = 0; i < omp_particles_count; ++i) {
        // Compute bounding box [inclusive-inclusive]
        int x_min = (int)roundf(omp_particles[i].location[0] - omp_particles[i].radius);
        int y_min = (int)roundf(omp_particles[i].location[1] - omp_particles[i].radius);
        int x_max = (int)roundf(omp_particles[i].location[0] + omp_particles[i].radius);
        int y_max = (int)roundf(omp_particles[i].location[1] + omp_particles[i].radius);
        // Clamp bounding box to image bounds
        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= omp_output_image.width ? omp_output_image.width - 1 : x_max;
        y_max = y_max >= omp_output_image.height ? omp_output_image.height - 1 : y_max;
        // Store data for every pixel within the bounding box that falls within the radius
        int x = x_min;
        int y = y_min;
#pragma omp parallel for private(x, y) schedule(dynamic,8)
        for (x = x_min; x <= x_max; ++x) {
            for (y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - omp_particles[i].location[0];
                const float y_ab = (float)y + 0.5f - omp_particles[i].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= omp_particles[i].radius) {
                    const unsigned int pixel_offset = y * omp_output_image.width + x;
                    // Offset into cpu_pixel_contrib buffers is index + histogram
                    // Increment cpu_pixel_contribs, so next contributor stores to correct offset
                    unsigned int storage_offset;
#pragma omp atomic capture
                    storage_offset = omp_pixel_contribs[pixel_offset]++;
                    storage_offset += omp_pixel_index[pixel_offset];
                    // Copy data to cpu_pixel_contrib buffers
                    memcpy(omp_pixel_contrib_colours + (4 * storage_offset), omp_particles[i].color, 4 * sizeof(unsigned char));
                    memcpy(omp_pixel_contrib_depth + storage_offset, &omp_particles[i].location[2], sizeof(float));
                }
            }
        }
    }

    // Pair sort the colours contributing to each pixel based on ascending depth
    for (int i = 0; i < omp_output_image.width * omp_output_image.height; ++i) {
        // Pair sort the colours which contribute to a single pigment
        cpu_sort_pairs_openmp(
            omp_pixel_contrib_depth,
            omp_pixel_contrib_colours,
            omp_pixel_index[i],
            omp_pixel_index[i + 1] - 1
        );
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation functions with the correct inputs
    // Note: Only validate_equalised_histogram() MUST be uncommented, the others are optional
    validate_pixel_index(omp_pixel_contribs, omp_pixel_index, omp_output_image.width, omp_output_image.height);
    validate_sorted_pairs(omp_particles, omp_particles_count, omp_pixel_index, omp_output_image.width, omp_output_image.height, omp_pixel_contrib_colours, omp_pixel_contrib_depth);
#endif    
}
void openmp_stage3() {
    memset(omp_output_image.data, 255, omp_output_image.width * omp_output_image.height * omp_output_image.channels * sizeof(unsigned char));

    // Order dependent blending into output image
    int i;
    unsigned int j;
#pragma omp parallel for private(i, j) schedule(static,8)
    for (i = 0; i < omp_output_image.width * omp_output_image.height; ++i) {
        for (j = omp_pixel_index[i]; j < omp_pixel_index[i + 1]; ++j) {
            // Blend each of the red/green/blue colours according to the below blend formula
            // dest = src * opacity + dest * (1 - opacity);
            const float opacity = (float)omp_pixel_contrib_colours[j * 4 + 3] / (float)255;
            omp_output_image.data[(i * 3) + 0] = (unsigned char)((float)omp_pixel_contrib_colours[j * 4 + 0] * opacity + (float)omp_output_image.data[(i * 3) + 0] * (1 - opacity));
            omp_output_image.data[(i * 3) + 1] = (unsigned char)((float)omp_pixel_contrib_colours[j * 4 + 1] * opacity + (float)omp_output_image.data[(i * 3) + 1] * (1 - opacity));
            omp_output_image.data[(i * 3) + 2] = (unsigned char)((float)omp_pixel_contrib_colours[j * 4 + 2] * opacity + (float)omp_output_image.data[(i * 3) + 2] * (1 - opacity));
            // cpu_pixel_contrib_colours is RGBA
            // cpu_output_image.data is RGB (final output image does not have an alpha channel!)
        }
    }

#ifdef VALIDATION
    // TODO: Uncomment and call the validation function with the correct inputs
    validate_blend(omp_pixel_index, omp_pixel_contrib_colours, &omp_output_image);
#endif    
}
void openmp_end(CImage* output_image) {
    // Store return value
    output_image->width = omp_output_image.width;
    output_image->height = omp_output_image.height;
    output_image->channels = omp_output_image.channels;
    memcpy(output_image->data, omp_output_image.data, omp_output_image.width * omp_output_image.height * omp_output_image.channels * sizeof(unsigned char));
    // Release allocations
    free(omp_pixel_contrib_depth);
    free(omp_pixel_contrib_colours);
    free(omp_output_image.data);
    free(omp_pixel_index);
    free(omp_pixel_contribs);
    free(omp_particles);
    // Return ptrs to nullptr
    omp_pixel_contrib_depth = 0;
    omp_pixel_contrib_colours = 0;
    omp_output_image.data = 0;
    omp_pixel_index = 0;
    omp_pixel_contribs = 0;
    omp_particles = 0;
}

void cpu_sort_pairs_openmp(float* keys_start, unsigned char* colours_start, const int first, const int last) {
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
        cpu_sort_pairs_openmp(keys_start, colours_start, first, j - 1);
        cpu_sort_pairs_openmp(keys_start, colours_start, j + 1, last);
    }
}