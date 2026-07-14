#include <iostream>
#include <cuda_runtime.h>
#include <csignal>
#include <cstdlib>


void* d_memory = nullptr;

// Signal handler function
void handleSignal(int signal) {
    if (signal == SIGINT) {
        // Free the allocated memory
        cudaFree(d_memory);
        std::exit(0); // Exit the program with success code
    }
}

int main(int argc, char** argv) {
     // Register the signal handler for SIGINT (Ctrl+C)
    std::signal(SIGINT, handleSignal);

    // Check if memory size is provided as an argument
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <memory_amount_in_MB> [context_usage_in_MB]" << std::endl;
        return 1;
    }

    // Parse the memory size argument
    float memory_size;
    try {
        memory_size = std::stof(argv[1]); // Convert string to unsigned long long
    } catch (const std::invalid_argument& e) {
        std::cerr << "Invalid memory size argument. Must be a number." << std::endl;
        return 1;
    }
    memory_size = memory_size * 1024 * 1024; // Convert to bytes

    float oversubscription_factor = 1.0;
    // get oversubscription factor from environment variable
    char* oversubscription_factor_env = std::getenv("OVERSUBSCRIPTION_FACTOR");
    if (oversubscription_factor_env != nullptr) {
        oversubscription_factor = std::stof(oversubscription_factor_env);
    } else {
        printf("OVERSUBSCRIPTION_FACTOR environment variable is not set. Using default value: %.3f\n",
                oversubscription_factor);
    }

    size_t free_size, total_size;
    cudaMemGetInfo(&free_size, &total_size);
    // size_t ctx_usage = total_size - free_size;
    // read from command line
    size_t ctx_usage = 0;
    if (argc > 2) {
        ctx_usage = std::stoul(argv[2]);
        printf("Context usage: %lu (%.3f MB)\n", ctx_usage, (float)ctx_usage / (1024 * 1024));
    } else {
        // default to 240MB
        ctx_usage = 240*1024*1024;
        printf("Context usage is not set. Using default value: %lu (%.3f MB)\n",
                ctx_usage, (float)ctx_usage / (1024 * 1024));
    }

    size_t request_size = (size_t) free_size - ctx_usage - memory_size / oversubscription_factor;

    // ceil to 2MB
    request_size = (request_size + 2*1024*1024 - 1) & ~(2*1024*1024 - 1);

    cudaError_t err = cudaMalloc(&d_memory, request_size);

    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    cudaMemGetInfo(&free_size, &total_size);
    printf("OVERSUBSCRIPTION_FACTOR: %.3f\n", oversubscription_factor);
    printf("Memory size: %.3f MB (%lu bytes, %.3f GB)\n",
            (float)memory_size / (1024 * 1024), (size_t)memory_size, (float)memory_size / (1024 * 1024 * 1024));
    printf("Free memory:  %.3f MB (%lu bytes, %.3f GB)\n",
            (float)free_size / (1024 * 1024), free_size, (float)free_size / (1024 * 1024 * 1024));

    printf("----------------------------------------\n\n");
    printf("Press Ctrl+C to exit.\n");

    while (true) {}

    return 0;
}