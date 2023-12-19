
#include "mlir/IR/BuiltinDialect.h"
#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "llvm/Support/raw_os_ostream.h"

#include <fstream>
#include <string>
#include <cassert>

using namespace mlir;

namespace mlir {
namespace ep2 {

const char* toString(MemType ty) {
  switch (ty) {
    case MemType::LMEM: return "LMEM";
    case MemType::CLS: return "CLS";
    case MemType::CTM: return "CTM";
    case MemType::IMEM: return "IMEM";
    case MemType::EMEM: return "EMEM";
    default: {
      assert(false && "Unsupported memtype");
      return nullptr;
    }
  }
}

void EmitFilesPass::runOnOperation() {
  auto module = getOperation();

  std::string basePath = module->getAttr("ep2.basePath").cast<StringAttr>().getValue().str();
  const CollectInfoAnalysis& info = getCachedAnalysis<CollectInfoAnalysis>().value();

  {
    std::ofstream fout_prog_hdr(basePath + "/prog_hdr.h");
    fout_prog_hdr << "#ifndef _PROG_HDR_H_\n";
    fout_prog_hdr << "#define _PROG_HDR_H_\n\n";
    fout_prog_hdr << "#include \"nfplib.h\"\n";
    fout_prog_hdr << "#include <nfp/mem_ring.h>\n\n";
    fout_prog_hdr << "__packed struct __wrapper_arg_t {\n";
    fout_prog_hdr << "\tint32_t f0;\n"; // atom
    fout_prog_hdr << "\tchar* f1;\n"; // ptr to event
    fout_prog_hdr << "};\n\n";
    
    // emit structs
    for (const auto& pr : info.structDefs) {
      fout_prog_hdr << (pr.second.isPacked() ? "__packed " : "") << "struct " << pr.first << " {\n";
      for (int i = 0; i<pr.second.getBody().size(); ++i) {
        mlir::Type ty = pr.second.getBody()[i];
        if (isa<LLVM::LLVMPointerType>(ty)) {
          // TODO assume buffer is the only ptr for now.
          fout_prog_hdr << "\tchar*" << " f" << i << ";\n";
        } else if (isa<mlir::IntegerType>(ty)) {
          fout_prog_hdr << "\tint" << cast<mlir::IntegerType>(ty).getWidth() << "_t f" << i << ";\n";
        }
      }
      if (pr.first.find("context_chain") != std::string::npos) {
        fout_prog_hdr << "\tint32_t ctx_id;\n";
      } else if (pr.first.find("event_param") != std::string::npos) {
        fout_prog_hdr << "\tstruct context_chain_1_t* ctx;\n";
      }

      fout_prog_hdr << "};\n\n";
    }

    // emit work queues
    int workq_id_incr = 10;
    for (const auto& q : info.eventQueues) {
      std::string eventName = q.first;

      fout_prog_hdr << "#define WORKQ_SIZE_" << eventName << " " << q.second.second << '\n';
      fout_prog_hdr << "#define WORKQ_ID_" << eventName << " " << (workq_id_incr++) << '\n';
      fout_prog_hdr << "#define WORKQ_TYPE_" << eventName << " " << "MEM_TYEP_" << toString(q.second.first) << '\n';
      fout_prog_hdr << toString(q.second.first) << "_WORKQ_DECLARE(workq_" << eventName << ", WORKQ_SIZE_" << eventName << ");\n\n";
    }

    // emit context allocation code
    std::string declType = "context_chain_1_t";
    std::string declId = "context_chain_pool";
    std::string declName = "context_chain_ring";
    std::string declSize = "2048";
    std::string declPlace = toString(MemType::EMEM);

    fout_prog_hdr << declPlace << "_CONTEXTQ_DECLARE(" << declType << ", " << declId << ", " << declSize << ");\n";
    fout_prog_hdr << "MEM_RING_INIT(" << declName << ", " << declSize << ");\n\n";
    fout_prog_hdr << "__forceinline static void init_" << declName << "() {\n";
    fout_prog_hdr << "\tunsigned int idx, rnum, raddr_hi, init_range;\n";
    fout_prog_hdr << "\tinit_range = IF_SIMULATION ? 10 : " << declSize << ";\n";
    fout_prog_hdr << "\tif (ctx() == 0) {\n";
    fout_prog_hdr << "\t\trnum = MEM_RING_GET_NUM(" << declName << ");\n";
    fout_prog_hdr << "\t\traddr_hi = MEM_RING_GET_MEMADDR(" << declName << ");\n";
    fout_prog_hdr << "\t\tfor (idx=1; idx<init_range; idx++) mem_ring_journal_fast(rnum, raddr_hi, idx);\n";
    fout_prog_hdr << "\t}\n";
    fout_prog_hdr << "\tfor (idx=0; idx<" << declSize << "; ++idx) " << declId << "[idx].ctx_id = idx;\n";
    fout_prog_hdr << "}\n\n";
    fout_prog_hdr << "__forceinline static struct " << declType << "* alloc_" << declName << "_entry() {\n";
    fout_prog_hdr << "\t__xread unsigned int context_idx;\n";
    fout_prog_hdr << "\tunsigned int rnum, raddr_hi;\n";
    fout_prog_hdr << "\trnum = MEM_RING_GET_NUM(" << declName << ");\n";
    fout_prog_hdr << "\traddr_hi = MEM_RING_GET_MEMADDR(" << declName << ");\n";
    fout_prog_hdr << "\twhile (mem_ring_get(rnum, raddr_hi, &context_idx, sizeof(context_idx)) != 0);\n";
    fout_prog_hdr << "\treturn &" << declId << "[context_idx];\n";
    fout_prog_hdr << "}\n\n";
    fout_prog_hdr << "#endif\n";
  }
  {
    std::unordered_map<std::string, func::FuncOp> nameToFunc;
    module->walk([&](func::FuncOp fop){
      nameToFunc[fop.getName().str()] = fop;
    });

    // each event is a stage in the pipeline.
    for (const auto& pr : info.eventAllocs) {
      std::string eventName = pr.first.substr(0, pr.first.find("_a_"));
      std::string atomName = pr.first.substr(pr.first.find("_a_") + 3);
      std::string funcName = "__event___handler_" + eventName + "_" + atomName;

      std::ofstream fout_stage(basePath + "/" + atomName + ".c");
      fout_stage << "#include \"nfplib.h\"\n";
      fout_stage << "#include \"prog_hdr.h\"\n\n";

      for (const auto& localAlloc : pr.second) {
        fout_stage << "struct " << localAlloc.first << " " << localAlloc.second << ";\n";
      }

      fout_stage << "struct event_param_" << eventName << " work;\n";
      fout_stage << "__xrw struct event_param_" << eventName << " work_ref;\n";
      fout_stage << "struct __wrapper_arg_t wrap_in;\n";

      bool isLastStage = info.eventDeps.find(eventName) == info.eventDeps.end();

      if (!isLastStage) {
        std::string nextEventName = info.eventDeps.find(eventName)->second;
        fout_stage << "struct event_param_" << nextEventName << " next_work;\n";
        fout_stage << "__xrw struct event_param_" << nextEventName << " next_work_ref;\n";
        fout_stage << "struct __wrapper_arg_t wrap_out;\n\n";
      }

      {
        llvm::raw_os_ostream func_stage(fout_stage);
        auto tRes = emitc::translateToCpp(nameToFunc[funcName], func_stage, true);
        assert(tRes.succeeded());
      }

      fout_stage << "\nint main(void) {\n";

      bool isFirstStage = true;
      for (const auto& pr : info.eventDeps) {
        if (pr.second == eventName) {
          isFirstStage = false;
        }
      }

      if (isFirstStage) {
        fout_stage << "\tinit_context_chain_ring();\n";
      }

      fout_stage << "\tinit_recv_event_workq(WORKQ_ID_" << eventName << ", workq_" << eventName << ", WORKQ_TYPE_" << eventName << ", WORKQ_SIZE_" << eventName << ", 8);\n";
      fout_stage << "\tfor (;;) {\n";
      fout_stage << "\t\tcls_workq_add_thread(" << "WORKQ_ID_" << eventName << ", &work_ref, sizeof(work_ref));\n";
      fout_stage << "\t\twork = work_ref;\n";
      fout_stage << "\t\twrap_in.f1 = &work;\n";

      if (isFirstStage) {
        fout_stage << "\t\twork.ctx = alloc_context_chain_ring_entry();\n";
      }

      fout_stage << "\t\t" << funcName << "(&wrap_in";
      if (!isLastStage) {
        fout_stage << ", &wrap_out";
      }
      fout_stage << ");\n";

      // rely on invariant one event type can only EVER flow to one other (many atom types possibly).
      if (!isLastStage) {
        std::string nextEventName = info.eventDeps.find(eventName)->second;
        fout_stage << "\t\tnext_work_ref = next_work;\n";
        fout_stage << "\t\tcls_workq_add_work(WORKQ_ID_" << nextEventName << ", &next_work_ref, sizeof(next_work_ref));\n";
      }
      fout_stage << "\t}\n";
      fout_stage << "}\n";
    }
  }
}

}
}
