#include "ep2/dialect/Dialect.h"
#include "ep2/dialect/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/EquivalenceClasses.h"

namespace mlir {
namespace ep2 {

namespace {
  struct BufferOp {
    int offset;
    int size;
    enum {
      Emit, Extract, Tail
    } type = Emit;
    int end() const { return offset + size; }
    bool operator <(const BufferOp &rhs) const {
      return offset < rhs.offset;
    }
    bool operator==(const BufferOp &rhs) const {
      return std::tie(offset, size) == std::tie(rhs.offset, rhs.size);
    }
    bool operator!=(const BufferOp &rhs) const { return !(*this == rhs); }
  };

  struct BufferHistory {
    bool known;
    std::vector<BufferOp> ops;
    int cur_offset = 0;

    BufferHistory(bool known = false, std::vector<BufferOp> ops = {})
      : known(known), ops(std::move(ops)) { }

    void operate(int size, bool emit = false) {
        insert({cur_offset, size});
    }

    bool insert(const BufferOp op) {
      if (known) {
        auto it = std::lower_bound(ops.begin(), ops.end(), op);
        ops.insert(it, op);
        if (op.end() > it->offset)
          known = false;
      }
      return known;
    }


    bool operator==(const BufferHistory &rhs) const {
      if (!known || !rhs.known)
        return false;
      if (ops.size() != rhs.ops.size())
        return false;
      for (size_t i = 0; i < ops.size(); ++i)
        if (ops[i] != rhs.ops[i])
          return false;
      return true;
    }
    BufferHistory operator+(const BufferHistory &rhs) const {
      BufferHistory ret{known && rhs.known};
      if (!ret.known)
        return ret;

      ret.ops = std::move(ops);
      for (auto &op : rhs.ops) {
        ret.insert(op);
        if (!ret.known)
          break;
      }
      return ret;
    }
  };
  
  struct HistoryTable {
    llvm::DenseMap<Value, BufferHistory> table;

    void declare(Value value) {
      table.try_emplace(value, BufferHistory{true, {}});
    }
    void emit(Value value, Type type) {
      auto size = 10;
      table[value].operate(size);
    }
    void installMapping(std::map<Value, Value> &mapping) {
      // pass
    }
    HistoryTable &operator+=(const HistoryTable &rhs) {
      return *this;
    }
  };

  void analysisBlock(Block &block, HistoryTable &table) {
    for (auto &op : block) {
        TypeSwitch<Operation *, void>(&op)
          .Case<mlir::ep2::InitOp>([&](InitOp op) {
            if (op.getType().isa<BufferType>())
              table.declare(op);
          })
          .Case<mlir::ep2::EmitOp>([&](EmitOp op) {
            table.emit(op.getBuffer(), op.getValue().getType());
          })
          .Case<mlir::ep2::ExtractOp>([&](auto op) {
            table.emit(op.getBuffer(), op.getType());
          })
          .Case<scf::IfOp>([&](scf::IfOp op) {
            auto elseTable = table; // copy construct another table
            analysisBlock(op.getThenRegion().front(), table);
            analysisBlock(op.getElseRegion().front(), elseTable);
            table += elseTable;
          })
          .Default([&](auto op) {});
    }
  }
} // local namespace

BufferAnalysis::BufferAnalysis(Operation* module) {
}

} // namespace ep2
} // namespace mlir