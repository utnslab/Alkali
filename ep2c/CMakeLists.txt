set(LLVM_LINK_COMPONENTS
  Support
  )

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
set(MLIR_LINK_COMPONENTS
  ${dialect_libs}
  ${conversion_libs}
  ${translation_libs}
  ${extension_libs}

  MLIRAnalysis
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRExecutionEngine
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation
  MLIRMemRefDialect
  MLIRParser
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIROptLib
)

# ep2c
add_llvm_executable(ep2c
  ep2c.cpp
  AST.cpp
  MLIRGen.cpp

  DEPENDS
  MLIREP2
  )
llvm_update_compile_flags(ep2c)
target_link_libraries(ep2c PRIVATE ${MLIR_LINK_COMPONENTS})



