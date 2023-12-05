add_mlir_dialect_library(MLIREP2
        Dialect.cpp

        # Analysis
        LowerStructAnalysis.cpp
        HandlerDependencyAnalysis.cpp
        ContextBufferizationAnalysis.cpp
        
        # Passes
        NopEliminationPass.cpp
        ContextRefTypeAssignPass.cpp
        EmitFPGA.cpp
        AtomAnalysis.cpp
        ContextTypeInferencePass.cpp
        LowerEmitcPass.cpp
        CollectHeaderPass.cpp
        EmitFilesPass.cpp
        LowerIntrinsicsPass.cpp
        AtomAnalysis.cpp
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
