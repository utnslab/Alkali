
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

/*
We assume in previous passes, fields are appropriately aligned to enable optimizations.
Now, we actually insert padding in the struct declarations.
For now, no rearranging elements in compiler-defined structures like context.
*/
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
  auto isPow2 = [](unsigned x) {
    return (x & -x) == x;
  };

  auto module = getOperation();

  if (basePathOpt.empty()) {
    emitError(module.getLoc(), "basePath not specified");
    signalPassFailure();
    return;
  }

  std::optional<int> recvBufOffs;
  std::string basePath = basePathOpt.getValue();
  const CollectInfoAnalysis& info = getCachedAnalysis<CollectInfoAnalysis>().value();

  //  Extract different fields from handler name
  auto extractEventName = [&](std::string str) {
    return str.substr(0, str.find("_a_"));
  };
  auto extractAtomName = [&](std::string str) {
    int first = str.find("_a_");
    return str.substr(first + 3, str.find("_a_", first + 3) - first - 3);
  };
  auto extractHandlerName = [&](std::string str) {
    int first = str.find("_a_");
    int second = str.find("_a_", first + 3);
    return "__event_" + str.substr(second + 3);
  };
  auto makeFileName = [&](std::string str) {
    return extractAtomName(str) + "_" + str.substr(str.rfind("_") + 1);
  };

  // Emit header file
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
      fout_prog_hdr << "\tuint8_t storage[" << (width/8) << "];\n";
      fout_prog_hdr << "} uint48_t;\n\n";
    }

    fout_prog_hdr << "__packed struct __buf_t {\n";
    fout_prog_hdr << "\tchar* buf;\n";
    fout_prog_hdr << "\tunsigned offs;\n";
    fout_prog_hdr << "\tunsigned sz;\n";
    fout_prog_hdr << "};\n\n";
    
    std::unordered_map<std::string, int> sizeMap;

    // Emit structs
    for (const auto& pr : info.structDefs) {
      fout_prog_hdr << (pr.second.isPacked() ? "__packed " : "") << "struct " << pr.first << " {\n";

      bool isContext = pr.first.find("context_chain") != std::string::npos;
      bool isEvent = pr.first.find("event_param") != std::string::npos;

      std::vector<std::pair<int, unsigned>> padding = calcPadding(pr.second, isContext || isEvent);
      unsigned padPos = 0;
      unsigned sz = calcSize(pr.second);
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
          fout_prog_hdr << "\tuint" << cast<mlir::IntegerType>(ty).getWidth() << "_t f" << i << ";\n";
        }
        if (padPos < padding.size() && padding[padPos].first == i) {
          fout_prog_hdr << "\tuint8_t pad" << padPos << "[" << padding[padPos].second << "];\n";
          sz += padding[padPos].second;
          padPos += 1;
        }
      }
      if (isEvent) {
        if (sz % 8 != 0) {
          fout_prog_hdr << "\tuint8_t pad" << padPos << "[" << ((8 - (sz % 8)) % 8) << "];\n";
          padPos += 1;
        }
        if (pr.first == "event_param_NET_RECV") fout_prog_hdr << "\tstruct recv_meta_t meta;\n";
        if (pr.first == "event_param_NET_SEND") fout_prog_hdr << "\tstruct send_meta_t meta;\n";
        fout_prog_hdr << "\t__shared __cls struct context_chain_1_t* ctx;\n";
      }
      fout_prog_hdr << "};\n\n";
      sizeMap[pr.first] = sz;
    }

    auto nextPow2 = [](unsigned x) {
      x -= 1;
      x |= (x >> 1);
      x |= (x >> 2);
      x |= (x >> 4);
      x |= (x >> 8);
      x |= (x >> 16);
      return x+1;
    };

    // Emit work queues
    int workq_id_incr = 5;
    for (const auto& q : info.eventQueues) {
      std::string eventName = q.first;
      int sz = sizeMap[std::string{"event_param_"} + eventName];

      fout_prog_hdr << "#define WORKQ_SIZE_" << eventName << " " << nextPow2(sz * q.second.size) << '\n';
      fout_prog_hdr << "#define WORKQ_TYPE_" << eventName << " " << "MEM_TYEP_" << toStringDecl(q.second.memType) << '\n';

      for (int replica : q.second.replicas) {
        fout_prog_hdr << "#define WORKQ_ID_" << eventName << "_" << (replica+1) << " " << (workq_id_incr++) << '\n';
        fout_prog_hdr << toStringDecl(q.second.memType) << "_WORKQ_DECLARE(workq_" << eventName << "_" << (replica+1) << ", WORKQ_SIZE_" << eventName << ");\n\n";
      }
    }

    // Emit tables
    unsigned tableCtr = 0;
    for (const auto& pr : info.tableInfos) {
      const auto& tInfo = pr.second.first;
      fout_prog_hdr << "__packed " << tInfo.tableType << " {\n";
      fout_prog_hdr << "\t" << tInfo.valType << " table[" << tInfo.size << "];\n";
      fout_prog_hdr << "};\n";

      for (const std::string& field : pr.second.second) {
        if (tInfo.isLocal) {
          fout_prog_hdr << "__shared __lmem ";
        } else {
          fout_prog_hdr << "__export __shared __cls ";
        }
        fout_prog_hdr << tInfo.tableType << " " << field << ";\n";
        if (tInfo.size > 16) {
          fout_prog_hdr << "__shared __lmem struct flowht_entry_t " << field << "_index[" << tInfo.size << "];\n";
        }
      }
      fout_prog_hdr << "\n";
    }

    // emit context allocation code
    std::string declType = "context_chain_1_t";
    std::string declId = "context_chain_pool";
    std::string declName = "context_chain_ring";
    int declSize = 128;
    std::string declPlace = toStringDecl(MemType::CLS);

    fout_prog_hdr << declPlace << "_CONTEXTQ_DECLARE(" << declType << ", " << declId << ", " << declSize << ");\n";
    fout_prog_hdr << "#ifdef DO_CTXQ_INIT\n";
    fout_prog_hdr << "__export __shared __cls int " << declName << "_qHead = 0;\n";
    fout_prog_hdr << "#else\n";
    fout_prog_hdr << "__import __shared __cls int " << declName << "_qHead;\n";
    fout_prog_hdr << "#endif\n\n";

    assert(isPow2(declSize));
    fout_prog_hdr << "__forceinline static __shared __cls struct " << declType << "* alloc_" << declName << "_entry() {\n";
    fout_prog_hdr << "\t__xrw int context_idx = 1;\n";
    fout_prog_hdr << "\tcls_test_add(&context_idx, &" << declName << "_qHead, sizeof(context_idx));\n";
    fout_prog_hdr << "\treturn &" << declId << "[context_idx & " << (declSize-1)  << "];\n";
    fout_prog_hdr << "}\n\n";

    fout_prog_hdr << "__forceinline static struct __buf_t alloc_packet_buf() {\n";
    fout_prog_hdr << "\tstruct __buf_t buf;\n";
    fout_prog_hdr << "\tbuf.buf = alloc_packet_buffer();\n";
    fout_prog_hdr << "\tbuf.offs = 0;\n";
    fout_prog_hdr << "\tbuf.sz = 0;\n";
    fout_prog_hdr << "\treturn buf;\n";
    fout_prog_hdr << "}\n\n";

    fout_prog_hdr << "__forceinline static int hash(int x) {\n";
    fout_prog_hdr << "\treturn x;\n";
    fout_prog_hdr << "}\n\n";

    fout_prog_hdr << "#endif\n";
  }

  // Emit makefile
  {
    std::ofstream fout_makefile(basePath + "/Makefile");

    std::ifstream fin_prefix("./lib/ep2/MakefileHelpers/netronome.prefix");
    fout_makefile << fin_prefix.rdbuf() << "\n\n";

    unsigned ctr = 1;
    std::unordered_map<std::string, unsigned> atomToCtr;

    // Per ME, emit Makefile target
    for (const auto& pr : info.eventAllocs) {
      std::string atomName = makeFileName(pr.first);

      fout_makefile << "S" << ctr << "_SRCS := $(app_src_dir)/" << makeFileName(pr.first) << ".c\n";
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

    std::unordered_set<int> usedIslands;

    getOperation()->walk([&](func::FuncOp fop) {
      auto getIslandMEStr = [&](std::string instance) {
        int instance_id = std::stoi(instance.substr(2));
        int island_id = instance_id / 12;
        usedIslands.insert(island_id);
        int microengine_id = instance_id % 12;
        return "mei" + std::to_string(island_id) + ".me" + std::to_string(microengine_id);
      };
      
      // Emit mapping from makefile target to island/microengine location
      if (fop->hasAttr("location")) {
        fout_makefile << "\t-u " << getIslandMEStr(fop->getAttr("location").cast<mlir::StringAttr>().getValue().str()) << " -l $(S";
        fout_makefile << atomToCtr[fop->getAttr("atom").cast<mlir::StringAttr>().getValue().str() + "_" + fop.getName().str().substr(fop.getName().str().rfind("_") + 1)];
        fout_makefile << "_LIST) \\\n";
      }
    });
    for (int island : usedIslands) {
      fout_makefile << "\t-u mei" << island << ".me0 -l $(DMA_LIST) \\\n";
    }

    fout_makefile << "\t-u ila0.me0 -l $(ME_BLM_LIST) \\\n";
    fout_makefile << "\t-i i8 -e $(PICO_CODE)\n";

    std::ifstream fin_suffix("./lib/ep2/MakefileHelpers/netronome.suffix");
    fout_makefile << "\n\n" << fin_suffix.rdbuf();
  }
  // Emit C files
  {
    std::unordered_map<std::string, func::FuncOp> nameToFunc;
    module->walk([&](func::FuncOp fop){
      nameToFunc[fop.getName().str()] = fop;
    });

    std::unordered_set<std::string> firstVisited;

    for (const auto& pr : info.eventAllocs) {
      // Emit C file per ME.
      std::string eventName = extractEventName(pr.first);
      std::string atomName = extractAtomName(pr.first);
      std::string funcName = extractHandlerName(pr.first);
      int id = std::stoi(funcName.substr(funcName.rfind("_") + 1));

      bool isFirstStage = true;
      for (const auto& pr : info.eventDeps) {
        for (const auto& s : pr.second) {
          if (s == eventName) {
            isFirstStage = false;
          }
        }
      }
      
      std::string filePath = basePath + std::string{"/"} + makeFileName(pr.first) + ".c";
      std::ofstream fout_stage(filePath);

      // Ensure only one replica of first stage initializes global variables
      if (isFirstStage && firstVisited.find(eventName + atomName) == firstVisited.end()) {
        fout_stage << "#define DO_CTXQ_INIT\n\n";
        firstVisited.insert(eventName + atomName);
      }
      
      fout_stage << "#include \"nfplib.h\"\n";
      fout_stage << "#include \"prog_hdr.h\"\n";
      fout_stage << "#include \"extern/extern_dma.h\"\n";
      fout_stage << "#include \"extern/extern_net.h\"\n\n";

      for (const auto& localAlloc : pr.second) {
        fout_stage << "static struct " << localAlloc.first << " " << localAlloc.second << ";\n";
        fout_stage << "__xrw static struct " << localAlloc.first << " " << localAlloc.second << "_xfer;\n";
      }

      fout_stage << "static int rr_ctr = 0;\n";
      fout_stage << "__declspec(aligned(4)) struct event_param_" << eventName << " work;\n";
      fout_stage << "__xrw struct event_param_" << eventName << " work_ref;\n";

      bool isLastStage = info.eventDeps.find(eventName) == info.eventDeps.end();

      if (!isLastStage) {
        const auto& nextEventNames = info.eventDeps.find(eventName)->second;
        for (const std::string& nextEventName : nextEventNames) {
          fout_stage << "__declspec(aligned(4)) struct event_param_" << nextEventName << " next_work_" << nextEventName << ";\n";
          fout_stage << "__xrw struct event_param_" << nextEventName << " next_work_ref_" << nextEventName << ";\n";
        }
      }

      // Emit work dispatching functions, one per func. tell emitc backend which to call via an attribute.
      unsigned dispatchCtr = 0;
      mlir::Builder builder(&getContext());
      nameToFunc[funcName]->walk([&](emitc::CallOp callOp) {
        if (callOp.getCallee() == "__ep2_intrin_enq_work") {
          std::string event = (*callOp.getArgs())[0].cast<StringAttr>().getValue().str();
          std::string queueList = (*callOp.getArgs())[1].cast<StringAttr>().getValue().str();
          llvm::SmallVector<int> queues;
          
          auto parseDescription = [&](std::string info) {
            unsigned p = 0;
            while (p < info.size()) {
              int v = 0;
              while (p < info.size() && isdigit(info[p])) {
                v *= 10;
                v += (info[p] - '0');
                p += 1;
              }
              queues.push_back(v);
              while (p < info.size() && !isdigit(info[p])) {
                p += 1;
              }
            }
          };
          parseDescription(queueList);

          if (queues.size() < 2) {
            return;
          }

          dispatchCtr += 1;

          std::string field = (*callOp.getArgs())[2].cast<StringAttr>().getValue().str();
          fout_stage << "\n__forceinline static void dispatch" << dispatchCtr << " () {\n";
          if (field != "-1") {
            fout_stage << "\tswitch (hash(work.ctx->f" << field << ") % " << queues.size() << ") {\n";
          } else {
            fout_stage << "\tswitch (rr_ctr) {\n";
          }
          for (int i = 0; i<queues.size(); ++i) {
            fout_stage << "\tcase " << i << ":\n";
            fout_stage << "\t\tcls_workq_add_work(WORKQ_ID_" << event << "_" << (i+1) << ", &next_work_ref_" << event << ", sizeof(next_work_ref_" << event << "));\n";
            fout_stage << "\t\tbreak;\n";
          }
          fout_stage << "\t}\n";

          fout_stage << "\trr_ctr = ";
          if (isPow2(queues.size())) {
            fout_stage << "(rr_ctr + 1) & " << (queues.size()-1) << ";\n";
          } else {
            fout_stage << "rr_ctr == " << (queues.size()-1) << " ? 0 : (rr_ctr + 1);\n";
          }

          fout_stage << "}\n";
          callOp->setAttr("func", builder.getStringAttr(std::string{"dispatch"} + std::to_string(dispatchCtr)));
        }
      });

      {
        llvm::raw_os_ostream func_stage(fout_stage);
        fout_stage << "\n__forceinline\n";
        auto tRes = emitc::translateToCpp(nameToFunc[funcName], func_stage, true);
        assert(tRes.succeeded());
      }

      fout_stage << "\nint main(void) {\n";
      fout_stage << "\tinit_me_cam(16);\n";
      if (eventName != "NET_RECV") {
        fout_stage << "\tinit_recv_event_workq(WORKQ_ID_" << eventName << "_" << id << ", workq_" << eventName << "_" << id << ", WORKQ_TYPE_" << eventName << ", WORKQ_SIZE_" << eventName << ", 8);\n";
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
