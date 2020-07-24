#define K_THREADS 64
#define K_INDEX() ((gridDim.x * blockIdx.y + blockIdx.x) * blockDim.x + threadIdx.x)
#define RND() ((rand() & 0x7FFF) / float(0x8000))
#define ERRORCHECK() cErrorCheck(__FILE__, __LINE__)
