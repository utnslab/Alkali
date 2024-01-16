add_mlir_dialect_library(MLIREP2
        Dialect/Dialect.cpp
        Dialect/MemorySlotInstance.cpp

        # Analysis
        LowerStructAnalysis.cpp
        HandlerDependencyAnalysis.cpp
        ContextBufferizationAnalysis.cpp
        HandlerInOutAnalysis.cpp
        TableAnalysis.cpp
        AtomAnalysis.cpp
        BufferAnalysis.cpp
        
        # Passes
        ArchMappingPass.cpp
        ContextToArgumentPass.cpp
        CFToPredPass.cpp
        BufferToValuePass.cpp
        EmitFPGAHelper.cpp
        EmitFPGAController.cpp
        EmitFPGA.cpp
        EmitFPGATop.cpp
        HandlerInOutAnalysis.cpp
        TableAnalysis.cpp
        AtomAnalysis.cpp
        ContextTypeInferencePass.cpp
        LowerEmitcPass.cpp
        CollectHeaderPass.cpp
        EmitNetronomePass.cpp
        EP2LinearizePass.cpp
        LowerMemcpyPass.cpp
        AtomAnalysis.cpp
        StructUpdatePropagationPass.cpp
        LocalAllocAnalysis.cpp
        HandlerReplicationPass.cpp
        LowerNoctxswapPass.cpp

        Passes/ContextToMemPass.cpp
        Passes/BufferReusePass.cpp
        Passes/DeadFieldEliminatePass.cpp
        Passes/DeadParameterEliminatePass.cpp

        DEPENDS
        MLIREP2OpsIncGen
        EP2OpsAttrDefsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRTranslateLib
        MLIREmitCDialect
        MLIRTargetCpp
        MLIRFuncDialect
        )

# link to z3 libs
target_include_directories(MLIREP2 PRIVATE ${Z3_CXX_INCLUDE_DIRS})
target_link_libraries(MLIREP2 PRIVATE ${Z3_LIBRARIES})
target_compile_options(MLIREP2 PRIVATE ${Z3_COMPONENT_CXX_FLAGS})
