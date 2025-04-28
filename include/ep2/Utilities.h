#ifndef _EP2_UTILITIES_H_
#define _EP2_UTILITIES_H_

#include <vector>

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace ep2 {

struct OperatorRemoveGuard {
  std::vector<Operation *> ops{};
  ~OperatorRemoveGuard() { clear(); }
  bool clear() {
    auto empty = ops.empty();
    for (auto op : ops)
      op->erase();
    ops.clear();
    return !empty;
  }
  void add(Operation *op) { ops.push_back(op); }
  template <typename F>
  static void until(F &&f) {
    OperatorRemoveGuard guard;
    do { f(guard); } while (guard.clear());
  }
};

} // namespace ep2
} // namespace mlir

#endif // _EP2_UTILITIES_H_