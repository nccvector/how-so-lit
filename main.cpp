//
// Author: Faizan Ali (github.com/nccvector)
// Date: 23/05/20
//

#include "cuBuffer.h"
#include "optix_function_table_definition.h"

#include <iostream>

#include "launchParams.h"

extern "C" char embedded_ptx_code[];

struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void*                                        data;
};

/*! SBT record for a miss program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  void*                                        data;
};

/*! SBT record for a hitgroup program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord {
  __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  // just a dummy value - later examples will use more interesting
  // data here
  int                                          objectID;
};

// CUDA vars
CUcontext      cudaContext;
CUstream       stream;
cudaDeviceProp deviceProps;

// OptiX vars
OptixDeviceContext optixContext;


/*! @{ the pipeline we're building */
OptixPipeline               pipeline;
OptixPipelineCompileOptions pipelineCompileOptions = {};
OptixPipelineLinkOptions    pipelineLinkOptions    = {};

/*! @{ the module that contains out device programs */
OptixModule               module;
OptixModuleCompileOptions moduleCompileOptions = {};

/*! vector of all our program(group)s, and the SBT built around
    them */
std::vector<OptixProgramGroup> raygenPGs;
cuBuffer                       raygenRecordsBuffer;
std::vector<OptixProgramGroup> missPGs;
cuBuffer                       missRecordsBuffer;
std::vector<OptixProgramGroup> hitgroupPGs;
cuBuffer                       hitgroupRecordsBuffer;
OptixShaderBindingTable        sbt = {};

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

void OptixCreatePTXModule() {
  moduleCompileOptions.maxRegisterCount = 50;
  moduleCompileOptions.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  moduleCompileOptions.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

  pipelineCompileOptions                                  = {};
  pipelineCompileOptions.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  pipelineCompileOptions.usesMotionBlur                   = false;
  pipelineCompileOptions.numPayloadValues                 = 2;
  pipelineCompileOptions.numAttributeValues               = 2;
  pipelineCompileOptions.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
  pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

  pipelineLinkOptions.maxTraceDepth = 2;

  const std::string ptxCode = embedded_ptx_code;

  char   log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK( optixModuleCreateFromPTX( optixContext, &moduleCompileOptions, &pipelineCompileOptions, ptxCode.c_str(),
      ptxCode.size(), log, &sizeof_log, &module ) );

  std::cout << "03 - Successfully created PTX module" << std::endl;
}

void OptixCreateRaygenPrograms() {
  // we do a single ray gen program in this example:
  raygenPGs.resize( 1 );

  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc    pgDesc    = {};
  pgDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pgDesc.raygen.module               = module;
  pgDesc.raygen.entryFunctionName    = "__raygen__renderFrame";

  // OptixProgramGroup raypg;
  char   log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK( optixProgramGroupCreate( optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &raygenPGs[0] ) );

  std::cout << "04 - Successfully created raygen programs" << std::endl;
}

void OptixCreateMissPrograms() {
  // we do a single ray gen program in this example:
  missPGs.resize( 1 );

  OptixProgramGroupOptions pgOptions = {};
  OptixProgramGroupDesc    pgDesc    = {};
  pgDesc.kind                        = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pgDesc.miss.module                 = module;
  pgDesc.miss.entryFunctionName      = "__miss__radiance";

  // OptixProgramGroup raypg;
  char   log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK( optixProgramGroupCreate( optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &missPGs[0] ) );

  std::cout << "05 - Successfully created miss programs" << std::endl;
}

void OptixCreateHitgroupPrograms() {
  // for this simple example, we set up a single hit group
  hitgroupPGs.resize( 1 );

  OptixProgramGroupOptions pgOptions  = {};
  OptixProgramGroupDesc    pgDesc     = {};
  pgDesc.kind                         = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pgDesc.hitgroup.moduleCH            = module;
  pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pgDesc.hitgroup.moduleAH            = module;
  pgDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

  char   log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK( optixProgramGroupCreate( optixContext, &pgDesc, 1, &pgOptions, log, &sizeof_log, &hitgroupPGs[0] ) );

  std::cout << "06 - Successfully created hitgroup programs" << std::endl;
}

void OptixCreatePipeline() {
  std::vector<OptixProgramGroup> programGroups;
  for ( auto pg : raygenPGs ) {
    programGroups.push_back( pg );
  }
  for ( auto pg : missPGs ) {
    programGroups.push_back( pg );
  }
  for ( auto pg : hitgroupPGs ) {
    programGroups.push_back( pg );
  }

  char   log[2048];
  size_t sizeof_log = sizeof( log );
  OPTIX_CHECK( optixPipelineCreate( optixContext, &pipelineCompileOptions, &pipelineLinkOptions, programGroups.data(),
      (int) programGroups.size(), log, &sizeof_log, &pipeline ) );

  OPTIX_CHECK( optixPipelineSetStackSize( /* [in] The pipeline to configure the stack size for */
      pipeline,
      /* [in] The direct stack size requirement for direct
                    callables invoked from IS or AH. */
      2 * 1024,
      /* [in] The direct stack size requirement for direct
                    callables invoked from RG, MS, or CH.  */
      2 * 1024,
      /* [in] The continuation stack requirement. */
      2 * 1024,
      /* [in] The maximum depth of a traversable graph
                    passed to trace. */
      1 ) );

  std::cout << "07 - Successfully created pipeline" << std::endl;
}


void OptixCreateSBT() {
  // ------------------------------------------------------------------
  // build raygen records
  // ------------------------------------------------------------------
  std::vector<RaygenRecord> raygenRecords;
  for ( int i = 0; i < raygenPGs.size(); i++ ) {
    RaygenRecord rec;
    OPTIX_CHECK( optixSbtRecordPackHeader( raygenPGs[i], &rec ) );
    rec.data = nullptr; /* for now ... */
    raygenRecords.push_back( rec );
  }
  raygenRecordsBuffer.alloc_and_copy_to_device( raygenRecords );
  sbt.raygenRecord = raygenRecordsBuffer.d_pointer();

  // ------------------------------------------------------------------
  // build miss records
  // ------------------------------------------------------------------
  std::vector<MissRecord> missRecords;
  for ( int i = 0; i < missPGs.size(); i++ ) {
    MissRecord rec;
    OPTIX_CHECK( optixSbtRecordPackHeader( missPGs[i], &rec ) );
    rec.data = nullptr; /* for now ... */
    missRecords.push_back( rec );
  }
  missRecordsBuffer.alloc_and_copy_to_device( missRecords );
  sbt.missRecordBase          = missRecordsBuffer.d_pointer();
  sbt.missRecordStrideInBytes = sizeof( MissRecord );
  sbt.missRecordCount         = (int) missRecords.size();

  // ------------------------------------------------------------------
  // build hitgroup records
  // ------------------------------------------------------------------

  // we don't actually have any objects in this example, but let's
  // create a dummy one so the SBT doesn't have any null pointers
  // (which the sanity checks in compilation would complain about)
  int                         numObjects = 1;
  std::vector<HitgroupRecord> hitgroupRecords;
  for ( int i = 0; i < numObjects; i++ ) {
    int            objectType = 0;
    HitgroupRecord rec;
    OPTIX_CHECK( optixSbtRecordPackHeader( hitgroupPGs[objectType], &rec ) );
    rec.objectID = i;
    hitgroupRecords.push_back( rec );
  }
  hitgroupRecordsBuffer.alloc_and_copy_to_device( hitgroupRecords );
  sbt.hitgroupRecordBase          = hitgroupRecordsBuffer.d_pointer();
  sbt.hitgroupRecordStrideInBytes = sizeof( HitgroupRecord );
  sbt.hitgroupRecordCount         = (int) hitgroupRecords.size();

  std::cout << "08 - Successfully created SBT" << std::endl;
}

LaunchParams launchParams;
cuBuffer     launchParamsBuffer;

void render() {
  // sanity check: make sure we launch only after first resize is
  // already done:
  if ( launchParams.fbSize.x == 0 ) {
    return;
  }

  launchParamsBuffer.copy_to_device( &launchParams, 1 );

  OPTIX_CHECK( optixLaunch( /*! pipeline we're launching launch: */
      pipeline, stream,
      /*! parameters and SBT */
      launchParamsBuffer.d_pointer(), launchParamsBuffer.size_in_bytes, &sbt,
      /*! dimensions of the launch: */
      launchParams.fbSize.x, launchParams.fbSize.y, 1 ) );
  // sync - make sure the frame is rendered before we download and
  // display (obviously, for a high-performance application you
  // want to use streams and double-buffering, but for this simple
  // example, this will have to do)
  CUDA_SYNC_CHECK();

  std::cout << "Rendered successfully" << std::endl;
}

cuBuffer colorBuffer;

void resize( const int2& newSize ) {
  // if window minimized
  if ( newSize.x == 0 | newSize.y == 0 ) {
    return;
  }

  // resize our cuda frame buffer
  colorBuffer.resize( newSize.x * newSize.y * sizeof( uint32_t ) );

  // update the launch parameters that we'll pass to the optix
  // launch:
  launchParams.fbSize      = newSize;
  launchParams.colorBuffer = (uint32_t*) colorBuffer.d_ptr;
}

int main() {

  // Allocate launchParams
  resize( make_int2( 800, 600 ) );

  OptixInit();
  OptixCreateContext();
  OptixCreatePTXModule();
  OptixCreateRaygenPrograms();
  OptixCreateMissPrograms();
  OptixCreateHitgroupPrograms();
  OptixCreatePipeline();
  OptixCreateSBT();

  // Initialize launchParamsBuffer
  launchParamsBuffer.alloc( sizeof( launchParams ) );

  render();

  return 0;
}
