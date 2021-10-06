// RUN: %run_test hipify "%s" "%t" %hipify_args 1 --experimental %clang_args

// CHECK: #include <hip/hip_runtime.h>
#include <stdio.h>

#define NUM_WORK_ELEMENTS (512)
#define WORK_BUFFER_SIZE (NUM_WORK_ELEMENTS * sizeof(float))
#define NUM_NODES NUM_WORK_ELEMENTS
#define NUM_FRAMES 1000

typedef struct {
  int index;
  float adder;
} node_t;

typedef struct {
  node_t* vector_table[NUM_NODES];
  node_t nodes[NUM_NODES];
} graph_control_t;

float* host_in_p;
float* host_out_p;
graph_control_t* host_gc_p;
float* dev_in_p;
float* dev_out_p;
graph_control_t* dev_gc_p;
int global_index[NUM_NODES] = {0};

// CHECK: hipGraph_t graph;
// CHECK-NEXT: hipGraphExec_t graphExec;
// CHECK-NEXT : hipGraphExec_t instance;
// CHECK-NEXT : hipStream_t stream;
// CHECK-NEXT : hipGraphNode_t node[NUM_NODES] = { 0 };
cudaGraph_t graph;
cudaGraphExec_t graphExec;
cudaGraphExec_t instance;
cudaStream_t stream;
cudaGraphNode_t node[NUM_NODES] = { 0 };

__global__
void add(const void* index,
         const graph_control_t* gc_p,
         const float *a,
         float* b) {
  int node_index = (long)index;
  float adder;
  int i = threadIdx.x;
  node_t* node_p =  gc_p->vector_table[node_index];
  node_index = node_p->index;
  adder = node_p->adder;
  if (i == node_index) {
    for (int j = 0; j < 100; j++)
      b[i] = a[i] + adder + node_index;
  }
}

void init(void) {
  int i = 0;
  // CHECK: hipStreamCreate(&stream);
  cudaStreamCreate(&stream);
  host_in_p = (float*) malloc(sizeof(float) * WORK_BUFFER_SIZE);
  host_out_p = (float*) malloc(sizeof(float) * WORK_BUFFER_SIZE);
  host_gc_p = (graph_control_t*) malloc(sizeof(graph_control_t));
  for (i = 0; i < WORK_BUFFER_SIZE; ++i) {
    host_in_p[i] = 1;
  }
  for (i = 0; i < WORK_BUFFER_SIZE; ++i) {
    host_out_p[i] = 42;
  }
  // CHECK: hipMalloc(&dev_in_p, WORK_BUFFER_SIZE);
  // CHECK-NEXT: hipMalloc(&dev_out_p, WORK_BUFFER_SIZE);
  // CHECK-NEXT: hipMalloc(&dev_gc_p, sizeof(graph_control_t));
  cudaMalloc(&dev_in_p, WORK_BUFFER_SIZE);
  cudaMalloc(&dev_out_p, WORK_BUFFER_SIZE);
  cudaMalloc(&dev_gc_p, sizeof(graph_control_t));
  for (i = 0; i < NUM_NODES; ++i) {
    host_gc_p->nodes[i].adder = 0;
    host_gc_p->nodes[i].index = i;
    host_gc_p->vector_table[i] = &(dev_gc_p->nodes[i]);
    global_index[i] = i;
  }
  // CHECK: if (hipGraphCreate(&graph, 0))
  if (cudaGraphCreate(&graph, 0))
    printf("Failed to create graph\n");

  /*  Create one long vertical graph with dependency between each element
      Would have parallel nodes and multiple instances of 
      it running in parallel if this was a real use-case,
      but since that is not the point here, lets skip that.
  */
  for (i = 0; i < NUM_NODES; ++i) {
    void* kargs[] = { &global_index[i],
                      &dev_gc_p,
                      &dev_in_p,
                      &dev_out_p };
    // CHECK: hipKernelNodeParams params = { .func           = (void*)add,
    cudaKernelNodeParams params = { .func           = (void*)add,
                                    .gridDim        = dim3(1, 1, 1),
                                    .blockDim       = dim3(NUM_WORK_ELEMENTS, 1 ,1),
                                    .sharedMemBytes = 0,
                                    .kernelParams   = kargs,
                                    .extra = NULL };
    // CHECK: if (hipGraphAddKernelNode(&node[i],
    if (cudaGraphAddKernelNode(&node[i],
                               graph,
                               0,
                               0,
                               &params))
      printf("Failed to create kernel node\n");
  }
  for (i = 0; i < (NUM_NODES - 1 ); ++i)
    // CHECK: hipGraphAddDependencies(graph,
    cudaGraphAddDependencies(graph,
                              &node[i],
                              &node[i+1],
                              1);
  // CHECK: hipGraphInstantiate(&graphExec,
  cudaGraphInstantiate(&graphExec,
                        graph,
                        NULL,
                        NULL,
                        0);
  /* Same input for dataplane every frame */
  // CHECK: hipMemcpy(dev_in_p,
  cudaMemcpy(dev_in_p,
             host_in_p,
             WORK_BUFFER_SIZE,
  // CHECK: hipMemcpyHostToDevice);
             cudaMemcpyHostToDevice);
}

void clean_up() {
  free(host_in_p);
  free(host_out_p);
  free(host_gc_p);
  // CHECK: hipFree(dev_in_p);
  // CHECK-NEXT: hipFree(dev_out_p);
  // CHECK-NEXT: hipFree(dev_gc_p);
  // CHECK-NEXT: hipDeviceReset();
  cudaFree(dev_in_p);
  cudaFree(dev_out_p);
  cudaFree(dev_gc_p);
  cudaDeviceReset();
}

void graph_launch() {
  // CHECK: hipGraphLaunch(graphExec, stream);
  cudaGraphLaunch(graphExec, stream);
}

/* Set up graphs, update input, execute and read out output once per frame */
int main (void) {
  int i = 0;
  /* Set up kernels (including parameters) and graphs once at init stage */
  init();
  /* Execute a bunch of frames */
  for (int framenbr = 0; framenbr < NUM_FRAMES; ++framenbr) {
    /* Set up control structure for one frame */
    for(i = 0; i < NUM_NODES; ++i) {
      host_gc_p->nodes[i].adder = framenbr * 1.0;
    }
    /* One copy for all data needed to execute one frame, queued in same stream as the rest */
    // CHECK: hipMemcpyAsync(dev_gc_p,
    cudaMemcpyAsync(dev_gc_p,
                    host_gc_p,
                    sizeof(graph_control_t),
    // CHECK: hipMemcpyHostToDevice,
                    cudaMemcpyHostToDevice,
                    stream);
    /* Kick graph */
    graph_launch();
    /*  One read-out for all data produced this frame, queued in same stream as the rest.   */
    // CHECK: hipMemcpyAsync(host_out_p,
    cudaMemcpyAsync(host_out_p,
                        dev_out_p,
                        WORK_BUFFER_SIZE,
    // CHECK: hipMemcpyDeviceToHost,
                        cudaMemcpyDeviceToHost,
                        stream);

    /*****************************************************************************************/
    /*  Here, use the offloaded CPU core to do things involved in setting up the control and */ 
    /*  data plane info for next frame                                                       */
    /*****************************************************************************************/

    /* If still not done, wait for current frame to finish */
    // CHECK: hipStreamSynchronize(stream);
    cudaStreamSynchronize(stream);
    /* Just some printouts to uncomment for quick sanity check of functional behaviour  */
    printf("\n\nOutput buffer, frame %d:\n", framenbr);
    for(i = 0; i < NUM_WORK_ELEMENTS; ++i) {
      printf("%f ", host_out_p[i]);
      if (i && i % 16 == 0)
        printf("\n");
    }
  }
  clean_up();
  return 0;
}
