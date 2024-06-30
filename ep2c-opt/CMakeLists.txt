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

add_llvm_executable(ep2c-opt
  ep2c-opt.cpp

  DEPENDS
  MLIREP2
  MLIRPolygeist
  )
llvm_update_compile_flags(ep2c-opt)
target_link_libraries(ep2c-opt PRIVATE ${MLIR_LINK_COMPONENTS})
target_link_libraries(ep2c-opt PRIVATE MLIRPolygeist)
