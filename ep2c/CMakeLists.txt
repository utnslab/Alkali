set(LLVM_LINK_COMPONENTS
  Support
  )

add_llvm_executable(ep2c
  ep2c.cpp
  AST.cpp


  MLIRGen.cpp
  Dialect.cpp

  DEPENDS
  MLIREP2OpsIncGen
  )

llvm_update_compile_flags(ep2c)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(ep2c
  PRIVATE
  ${dialect_libs}
  ${translation_libs}
  ${extension_libs}
  MLIRAnalysis
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

mlir_check_link_libraries(standalone-translate)
