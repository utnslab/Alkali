
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
#include <algorithm>
#include <optional>

#include "Utils.h"

using namespace mlir;

namespace mlir {
namespace ep2 {

static const char* toString(MemType ty) {
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

static unsigned calcSize(mlir::Type ty) {
  if (isa<mlir::IntegerType>(ty)) {
    return cast<mlir::IntegerType>(ty).getWidth()/8;
  } else if (isa<LLVM::LLVMPointerType>(ty)) {
    return 8;
  } else if (isa<LLVM::LLVMStructType>(ty)) {
    unsigned width = 0;
    for (mlir::Type eTy : cast<LLVM::LLVMStructType>(ty).getBody()) { 
      width += calcSize(eTy);
    }
    return width;
  }
};

// for now, no rearranging elements.
static std::vector<std::pair<int, unsigned>> calcPadding(const mlir::LLVM::LLVMStructType& ty, bool canOptimizeLayout) {
  std::vector<std::pair<int, unsigned>> paddingInfo;
  unsigned sz = calcSize(ty);

  auto moduloUp = [](unsigned v, unsigned m) {
    return (m - (v % m)) % m;
  };

  auto naturalAlign = [](unsigned sz) {
    if (sz <= 1) {
      return 1;
    } else if (sz == 2) {
      return 2;
    } else if (sz >= 3 || sz <= 4) {
      return 4;
    } else {
      return 8;
    }
  };

  if (canOptimizeLayout) {
    // space out members to ensure they are aligned naturally.
    // max out at 8-byte alignment.
    unsigned pos = 0;
    for (int i = 0; i<ty.getBody().size(); ++i) {
      auto ety = ty.getBody()[i];
      unsigned memberSz = calcSize(ety);
      if (memberSz % 4 != 0) {
        memberSz += moduloUp(memberSz, 4);
      }
      unsigned align = naturalAlign(memberSz);
      if (pos % align != 0) {
        unsigned pad = moduloUp(pos, align);
        pos += pad;
        paddingInfo.emplace_back(i-1, pad);
      }
      pos += memberSz;
    }
    sz = pos;
  }
  if (sz % 4 != 0) {
    paddingInfo.emplace_back(ty.getBody().size()-1, moduloUp(sz, 4));
  }
  return paddingInfo;
}

void EmitNetronomePass::runOnOperation() {
  auto module = getOperation();

  if (basePathOpt.empty()) {
    emitError(module.getLoc(), "basePath not specified");
    signalPassFailure();
    return;
  }

  std::optional<int> recvBufOffs;
  std::string basePath = basePathOpt.getValue();
  const CollectInfoAnalysis& info = getCachedAnalysis<CollectInfoAnalysis>().value();

  {
    std::ofstream fout_prog_hdr(basePath + "/prog_hdr.h");
    fout_prog_hdr << "#ifndef _PROG_HDR_H_\n";
    fout_prog_hdr << "#define _PROG_HDR_H_\n\n";
    fout_prog_hdr << "#include \"nfplib.h\"\n";
    fout_prog_hdr << "#include <nfp/mem_ring.h>\n";
    fout_prog_hdr << "#include \"extern/extern_net_meta.h\"\n\n";

    for (unsigned width : info.typeBitWidths) {
      if (width == 8 || width == 16 || width == 32 || width == 64) {
        continue;
      }
      fout_prog_hdr << "typedef __packed struct __int" << width << " {\n";
      fout_prog_hdr << "\tint8_t storage[" << (width/8) << "];\n";
      fout_prog_hdr << "} int48_t;\n\n";
    }

    fout_prog_hdr << "__packed struct __buf_t {\n";
    fout_prog_hdr << "\tchar* buf;\n";
    fout_prog_hdr << "\tunsigned offs;\n";
    fout_prog_hdr << "\tunsigned sz;\n";
    fout_prog_hdr << "};\n\n";
    
    // emit structs
    for (const auto& pr : info.structDefs) {
      fout_prog_hdr << (pr.second.isPacked() ? "__packed " : "") << "struct " << pr.first << " {\n";

      bool isContext = pr.first.find("context_chain") != std::string::npos;
      bool isEvent = pr.first.find("event_param") != std::string::npos;

      std::vector<std::pair<int, unsigned>> padding = calcPadding(pr.second, isContext || isEvent);
      unsigned padPos = 0;
      for (int i = 0; i<pr.second.getBody().size(); ++i) {
        mlir::Type ty = pr.second.getBody()[i];
        if (isa<LLVM::LLVMStructType>(ty)) {
          if (cast<LLVM::LLVMStructType>(ty).getName() == "__buf_t") {
            if (pr.first.find("NET_SEND") != std::string::npos) recvBufOffs = i;
            fout_prog_hdr << "\tstruct __buf_t" << " f" << i << ";\n";
          } else {
            fout_prog_hdr << "\tstruct " << cast<LLVM::LLVMStructType>(ty).getName().str() << " f" << i << ";\n";
          }
        } else if (isa<mlir::IntegerType>(ty)) {
          fout_prog_hdr << "\tint" << cast<mlir::IntegerType>(ty).getWidth() << "_t f" << i << ";\n";
        }
        if (padPos < padding.size() && padding[padPos].first == i) {
          fout_prog_hdr << "\tint8_t pad" << padPos << "[" << padding[padPos].second << "];\n";
          padPos += 1;
        }
      }
      if (isContext) {
        fout_prog_hdr << "\tint32_t ctx_id;\n";
      } else if (isEvent) {
        if (pr.first == "event_param_NET_RECV") fout_prog_hdr << "\tstruct recv_meta_t meta;\n";
        if (pr.first == "event_param_NET_SEND") fout_prog_hdr << "\tstruct send_meta_t meta;\n";
        fout_prog_hdr << "\tstruct context_chain_1_t* ctx;\n";
      }
      fout_prog_hdr << "};\n\n";
    }

    // emit work queues
    int workq_id_incr = 10;
    for (const auto& q : info.eventQueues) {
      std::string eventName = q.first;

      fout_prog_hdr << "#define WORKQ_SIZE_" << eventName << " " << q.second.size << '\n';
      fout_prog_hdr << "#define WORKQ_ID_" << eventName << " " << (workq_id_incr++) << '\n';
      fout_prog_hdr << "#define WORKQ_TYPE_" << eventName << " " << "MEM_TYEP_" << toString(q.second.memType) << '\n';
      fout_prog_hdr << toString(q.second.memType) << "_WORKQ_DECLARE(workq_" << eventName << ", WORKQ_SIZE_" << eventName << ");\n\n";
    }

    unsigned tableCtr = 0;
    for (const auto& tInfo : info.tableInfos) {
      fout_prog_hdr << "__packed " << tInfo.tableType << " {\n";
      fout_prog_hdr << "\t" << tInfo.valType << " table[" << tInfo.size << "];\n";
      fout_prog_hdr << "};\n";
      // TODO not lmem always
      fout_prog_hdr << "__shared " << tInfo.tableType << " " << tInfo.tableId << ";\n\n";
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
    fout_prog_hdr << "\tfor (idx=0; idx<init_range; ++idx) " << declId << "[idx].ctx_id = idx;\n";
    fout_prog_hdr << "}\n\n";
    fout_prog_hdr << "__forceinline static struct " << declType << "* alloc_" << declName << "_entry() {\n";
    fout_prog_hdr << "\t__xread unsigned int context_idx;\n";
    fout_prog_hdr << "\tunsigned int rnum, raddr_hi;\n";
    fout_prog_hdr << "\trnum = MEM_RING_GET_NUM(" << declName << ");\n";
    fout_prog_hdr << "\traddr_hi = MEM_RING_GET_MEMADDR(" << declName << ");\n";
    fout_prog_hdr << "\twhile (mem_ring_get(rnum, raddr_hi, &context_idx, sizeof(context_idx)) != 0);\n";
    fout_prog_hdr << "\treturn &" << declId << "[context_idx];\n";
    fout_prog_hdr << "}\n\n";
    fout_prog_hdr << "__forceinline static struct __buf_t alloc_packet_buf() {\n";
    fout_prog_hdr << "\tstruct __buf_t buf;\n";
    fout_prog_hdr << "\tbuf.buf = alloc_packet_buffer();\n";
    fout_prog_hdr << "\tbuf.offs = 0;\n";
    fout_prog_hdr << "\tbuf.sz = 0;\n";
    fout_prog_hdr << "\treturn buf;\n";
    fout_prog_hdr << "}\n\n";
    fout_prog_hdr << "#endif\n";
  }
  {
    std::ofstream fout_makefile(basePath + "/Makefile");

    std::ifstream fin_prefix("./lib/ep2/MakefileHelpers/netronome.prefix");
    fout_makefile << fin_prefix.rdbuf() << "\n\n";

    unsigned ctr = 1;
    std::unordered_map<std::string, unsigned> atomToCtr;

    for (const auto& pr : info.eventAllocs) {
      std::string atomName = pr.first.substr(pr.first.find("_a_") + 3);

      fout_makefile << "S" << ctr << "_SRCS := $(app_src_dir)/" << atomName << ".c\n";
      fout_makefile << "S" << ctr << "_LIST := " << atomName << ".list\n";
      fout_makefile << "S" << ctr << "_DEFS := $(mlir_NFCCFLAGS)\n";
      fout_makefile << "$(S" << ctr << "_LIST): $(mlir_NFCCSRCS) $(S" << ctr << "_SRCS)\n";
      fout_makefile << "\t@echo \"--- Building #@\"\n";
      fout_makefile << "\t$(Q) $(NFCC) $(S" << ctr << "_DEFS) -Fe$@ $(S" << ctr << "_SRCS) $(mlir_NFCCSRCS)\n";
      fout_makefile << "global_LIST_FILES += $(S" << ctr << "_LIST)\n";

      atomToCtr.emplace(atomName, ctr);
      ctr += 1;
    }

    fout_makefile << "core.fw: $(global_LIST_FILES)\n";
    fout_makefile << "\t@echo \"--- Linking $@\"\n";
    fout_makefile << "\t$(NFLD) $(mlir_NFLDFLAGS) \\\n";
    fout_makefile << "\t-elf $@ \\\n";

    getOperation()->walk([&](func::FuncOp fop) {
      auto getIslandMEStr = [](std::string instance) {
        std::string island = instance.substr(1, instance.find("cu")-1);
        std::string microEngine = instance.substr(instance.find("cu") + 2);
        return "mei" + island + ".me" + microEngine;
      };

      if (fop->hasAttr("instances")) {
        //atomToCtr[fop->getAttr("atom").cast<mlir::StringAttr>().getValue().str()]
        for (const auto& inst : cast<mlir::ArrayAttr>(fop->getAttr("instances")).getValue()) {
          fout_makefile << "\t-u " << getIslandMEStr(cast<mlir::StringAttr>(inst).getValue().str()) << " -l $(";
        }
      }
    });

    /*
    for (const auto& pr : placementInfo.placementMap) {
      fout_makefile << "\t-u mei" << pr.second.first << ".me" << pr.second.second << " -l $(";
      // TODO assumes all separate-file externs are DMA's, and only one use of DMA.
      if (atomToCtr.find(pr.first) == atomToCtr.end()) {
        fout_makefile << "DMA_LIST) \\\n";
      } else {
        fout_makefile << "S" << atomToCtr[pr.first] << "_LIST) \\\n";
      }
    }
    */

    fout_makefile << "\t-u ila0.me0 -l $(ME_BLM_LIST) \\\n";
    fout_makefile << "\t-i i8 -e $(PICO_CODE)\n";

    std::ifstream fin_suffix("./lib/ep2/MakefileHelpers/netronome.suffix");
    fout_makefile << "\n\n" << fin_suffix.rdbuf();
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
      
      // TODO add code to wait for initialization

      std::ofstream fout_stage(basePath + "/" + atomName + ".c");
      fout_stage << "#include \"nfplib.h\"\n";
      fout_stage << "#include \"prog_hdr.h\"\n";
      fout_stage << "#include \"extern/extern_dma.h\"\n";
      fout_stage << "#include \"extern/extern_net.h\"\n\n";

      for (const auto& localAlloc : pr.second) {
        fout_stage << "struct " << localAlloc.first << " " << localAlloc.second << ";\n";
        fout_stage << "__xrw struct " << localAlloc.first << " " << localAlloc.second << "_xfer;\n";
      }

      fout_stage << "__declspec(aligned(4)) struct event_param_" << eventName << " work;\n";
      fout_stage << "__xrw struct event_param_" << eventName << " work_ref;\n";

      bool isLastStage = info.eventDeps.find(eventName) == info.eventDeps.end();

      if (!isLastStage) {
        const std::vector<std::string>& nextEventNames = info.eventDeps.find(eventName)->second;
        for (const std::string& nextEventName : nextEventNames) {
          fout_stage << "__declspec(aligned(4)) struct event_param_" << nextEventName << " next_work_" << nextEventName << ";\n";
          fout_stage << "__xrw struct event_param_" << nextEventName << " next_work_ref_" << nextEventName << ";\n";
        }
      }

      {
        llvm::raw_os_ostream func_stage(fout_stage);
        fout_stage << "\n__forceinline\n";
        auto tRes = emitc::translateToCpp(nameToFunc[funcName], func_stage, true);
        assert(tRes.succeeded());
      }

      fout_stage << "\nint main(void) {\n";

      bool isFirstStage = true;
      for (const auto& pr : info.eventDeps) {
        for (const auto& s : pr.second) {
          if (s == eventName) {
            isFirstStage = false;
          }
        }
      }

      fout_stage << "\tinit_me_cam(16);\n";
      if (isFirstStage) {
        fout_stage << "\tinit_context_chain_ring();\n";
      }

      if (eventName != "NET_RECV") {
        fout_stage << "\tinit_recv_event_workq(WORKQ_ID_" << eventName << ", workq_" << eventName << ", WORKQ_TYPE_" << eventName << ", WORKQ_SIZE_" << eventName << ", 8);\n";
      }
      fout_stage << "\twait_global_start_();\n";

      fout_stage << "\tfor (;;) {\n";
      fout_stage << "\t\t" << funcName << "();\n";
      fout_stage << "\t}\n";
      fout_stage << "}\n";
    }
  }
}

}
}
