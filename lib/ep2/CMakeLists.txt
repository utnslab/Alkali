add_mlir_dialect_library(MLIREP2
        Dialect.cpp
        
        # Passes
        NopEliminationPass.cpp

        # ADDITIONAL_HEADER_DIRS
        # ${PROJECT_SOURCE_DIR}/include/Standalone

        DEPENDS
        MLIREP2OpsIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRFuncDialect
        )
