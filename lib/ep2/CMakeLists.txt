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
        LowerLLVMPass.cpp
        EmitLLVMHeaderPass.cpp
        HandlerReplicationPass.cpp
        LowerNoctxswapPass.cpp
        RepackStructTypesPass.cpp
        GprPromotionPass.cpp
        PipelineHandlerPass.cpp

        Passes/ContextToMemPass.cpp
        Passes/BufferReusePass.cpp
        Passes/DeadFieldEliminatePass.cpp
        Passes/DeadParameterEliminatePass.cpp
        Passes/CanonicalizePass.cpp
        Passes/AtomicIdentificationPass.cpp
        Passes/FPGABufferToStoragePass.cpp
        Passes/ControllerGenerationPass.cpp
        Passes/GlobalToPartitionPass.cpp

        # conversion and frontend
        Passes/Conversion/LiftLLVMPass.cpp
        Passes/Conversion/SplitFunction.cpp
        Passes/Conversion/LiftUtils.cpp

        Passes/Mapping/PerformanceModel.cpp
        Passes/Mapping/MappingUtils.cpp
        Passes/Mapping/PipelineCanonicalizePass.cpp

        DEPENDS
        MLIREP2OpsIncGen
        EP2OpsAttrDefsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRTranslateLib
        MLIREmitCDialect
        MLIRTargetCpp
        MLIRFuncDialect

        MLIRPolygeist
        )

# link to z3 libs
target_include_directories(MLIREP2 PRIVATE ${Z3_CXX_INCLUDE_DIRS})
target_link_libraries(MLIREP2 PRIVATE ${Z3_LIBRARIES})
target_compile_options(MLIREP2 PRIVATE ${Z3_COMPONENT_CXX_FLAGS})
