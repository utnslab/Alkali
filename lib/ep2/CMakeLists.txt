add_mlir_dialect_library(MLIREP2
        Dialect.cpp

        # Analysis
        LowerStructAnalysis.cpp
        HandlerDependencyAnalysis.cpp
        ContextAnalysis.cpp
        ContextBufferizationAnalysis.cpp
        
        # Passes
        NopEliminationPass.cpp
        LowerToLLVMPass.cpp
        FunctionRewritePass.cpp
        ContextRefTypeAssignPass.cpp
        AtomAnalysis.cpp
        ContextTypeInferencePass.cpp

        DEPENDS
        MLIREP2OpsIncGen
        EP2OpsAttrDefsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncDialect
        )
