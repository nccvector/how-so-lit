//
// Author: Faizan Ali (github.com/nccvector)
// Date: 23/05/20
//

#include "cuBuffer.h"
#include "optix_function_table_definition.h"

#include <iostream>

extern "C" char embedded_ptx_code[];

// CUDA vars
CUcontext      cudaContext;
CUstream       stream;
cudaDeviceProp deviceProps;

// OptiX vars
OptixDeviceContext optixContext;


/*! @{ the pipeline we're building */
OptixPipeline               pipeline;
OptixPipelineCompileOptions pipelineCompileOptions = {};
OptixPipelineLinkOptions    pipelineLinkOptions = {};

/*! @{ the module that contains out device programs */
OptixModule                 module;
OptixModuleCompileOptions   moduleCompileOptions = {};

void OptixInit() {
  // -------------------------------------------------------
  // check for available optix7 capable devices
  // -------------------------------------------------------
  cudaFree( 0 );
  int numDevices;
  cudaGetDeviceCount( &numDevices );
  if ( numDevices == 0 ) {
    throw std::runtime_error( "no CUDA capable devices found!" );
  }

  // -------------------------------------------------------
  // initialize optix
  // -------------------------------------------------------
  OPTIX_CHECK( optixInit() );

  std::cout << "01 - Successfully initialized optix" << std::endl;
}

void OptixCreateContext() {
  // for this sample, do everything on one device
  const int deviceID = 0;
  CUDA_CHECK( SetDevice( deviceID ) );
  CUDA_CHECK( StreamCreate( &stream ) );

  cudaGetDeviceProperties( &deviceProps, deviceID );

  CUresult cuRes = cuCtxGetCurrent( &cudaContext );
  if ( cuRes != CUDA_SUCCESS ) {
    fprintf( stderr, "Error querying current context: error code %d\n", cuRes );
  }

  OPTIX_CHECK( optixDeviceContextCreate( cudaContext, 0, &optixContext ) );

  std::cout << "02 - Successfully created optix context" << std::endl;
}

//void OptixCreatePTXModule() {
//  moduleCompileOptions.maxRegisterCount  = 50;
//  moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
//  moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
//
//  pipelineCompileOptions = {};
//  pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
//  pipelineCompileOptions.usesMotionBlur     = false;
//  pipelineCompileOptions.numPayloadValues   = 2;
//  pipelineCompileOptions.numAttributeValues = 2;
//  pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
//  pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
//
//  pipelineLinkOptions.maxTraceDepth          = 2;
//
//  const std::string ptxCode = embedded_ptx_code;
//
//  char log[2048];
//  size_t sizeof_log = sizeof( log );
//  OPTIX_CHECK(optixModuleCreateFromPTX(optixContext,
//      &moduleCompileOptions,
//      &pipelineCompileOptions,
//      ptxCode.c_str(),
//      ptxCode.size(),
//      log,&sizeof_log,
//      &module
//      ));
//}

int main() {
  OptixInit();
  OptixCreateContext();

  return 0;
}
