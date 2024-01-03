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
        
        # Passes
        ContextToArgumentPass.cpp
        CFToPredPass.cpp
        BufferToValuePass.cpp
        EmitFPGAHelper.cpp
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
