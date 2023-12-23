#include "ep2/dialect/Dialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"

using namespace mlir;
using namespace mlir::ep2;

//===----------------------------------------------------------------------===//
// Memory reinit/emit_value/extract_value
//===----------------------------------------------------------------------===//

// ReRefOps
llvm::SmallVector<MemorySlot> ReRefOp::getPromotableSlots() {
  return { MemorySlot{getResult(), getType()} };
}
Value ReRefOp::getDefaultValue(const MemorySlot &slot, RewriterBase &rewriter) {
  // since we do re-import, we return the original op
  return getInput();
}
void ReRefOp::handlePromotionComplete(const MemorySlot &slot,
                                      Value defaultValue,
                                      RewriterBase &rewriter) {
  // we handle the job of removing the original value to other passes
  rewriter.eraseOp(*this);
}
void ReRefOp::handleBlockArgument(const MemorySlot &slot,
                                  BlockArgument argument,
                                  RewriterBase &rewriter) {}

void rewriteNextInit(RewriterBase &rewriter, Value value) {
  // from the doc, "The rewriter is located after the promotable operation on call"
  // We take use of this to find our inserted operation and replace its value
  // find the next init value, and replace it's return value with 
  auto it = rewriter.getInsertionPoint();
  // TODO(zhiyuang): Check this. Will we see an "end"?
  while (!isa<InitOp>(*it)) it++;
  auto initOp = cast<InitOp>(*it);
  rewriter.replaceAllUsesWith(initOp.getResult(), value);
}

/// EmitOp's Memory Slot Interaface
bool EmitOp::loadsFrom(const MemorySlot &slot) {
  return getBuffer() == slot.ptr;
}
bool EmitOp::storesTo(const MemorySlot &slot) {
  return getBuffer() == slot.ptr;
}
Value EmitOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  auto initOp = rewriter.create<InitOp>(getLoc(), rewriter.getType<BufferType>());
  auto newOp = rewriter.create<EmitValueOp>(getLoc(), rewriter.getType<BufferType>(), initOp, getValue());
  return newOp.getResult();
}
// TODO(zhiyuang): check this. when do they use this call?
bool EmitOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getBuffer() == slot.ptr;
}
DeletionKind EmitOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition) {
  rewriteNextInit(rewriter, reachingDefinition);
  return DeletionKind::Delete;
}

/// ExtractOp's Memory Slot Interaface
bool ExtractOp::loadsFrom(const MemorySlot &slot) {
  return getBuffer() == slot.ptr;
}
bool ExtractOp::storesTo(const MemorySlot &slot) {
  return getBuffer() == slot.ptr;
}
Value ExtractOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  auto initOp = rewriter.create<InitOp>(getLoc(), rewriter.getType<BufferType>());
  auto newOp = rewriter.create<ExtractValueOp>(
      getLoc(), TypeRange{rewriter.getType<BufferType>(), getResult().getType()},
      initOp);
  rewriter.replaceAllUsesWith(getResult(), newOp.getResult(1));
  return newOp.getResult(0);
}
bool ExtractOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getBuffer() == slot.ptr;
}
DeletionKind ExtractOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition) {
  rewriteNextInit(rewriter, reachingDefinition);
  return DeletionKind::Delete;
}

/// DeRef's Memory Slot Interaface
bool DeRefOp::loadsFrom(const MemorySlot &slot) {
  return getInput() == slot.ptr;
}
bool DeRefOp::storesTo(const MemorySlot &slot) { return false; }
Value DeRefOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  llvm_unreachable("getStored should not be called on DeRefOp");
}
// TODO(zhiyuang): check this. when do they use this call?
bool DeRefOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  return blockingUse == slot.ptr && getInput() == slot.ptr;
}
DeletionKind DeRefOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition) {
  // `canUsesBeRemoved` checked this blocking use must be the loaded slot
  // pointer.
  rewriter.replaceAllUsesWith(getResult(), reachingDefinition);
  return DeletionKind::Delete;
}

//===----------------------------------------------------------------------===//
// Memory Load/Store/ContextRef
//===----------------------------------------------------------------------===//

llvm::SmallVector<MemorySlot> ContextRefOp::getPromotableSlots() {
  if (getOperation()->hasAttr("transferToValue"))
    return { MemorySlot{getResult(), getType().getValueType()} };
  else
    return {};
}

Value ContextRefOp::getDefaultValue(const MemorySlot &slot, RewriterBase &rewriter) {
  auto funcOp = getOperation()->getParentOfType<FuncOp>();
  for (unsigned i = 0; i < funcOp.getNumArguments(); i++) {
    auto argAttr = funcOp.getArgAttrOfType<StringAttr>(i, "ep2.context_name");
    if (argAttr && argAttr.getValue() == getName())
      return funcOp.getArgument(i);
  }
  // TODO(zhiyuang): add assert or warning here?
  return rewriter.create<InitOp>(getLoc(), slot.elemType);
}

void ContextRefOp::handlePromotionComplete(const MemorySlot &slot,
                                               Value defaultValue,
                                               RewriterBase &rewriter) {
  if (defaultValue.use_empty() && defaultValue.getDefiningOp())
    rewriter.eraseOp(defaultValue.getDefiningOp());
  rewriter.eraseOp(*this);
}

// TODO(zhiyuang): check this
void ContextRefOp::handleBlockArgument(const MemorySlot &slot,
                                           BlockArgument argument,
                                           RewriterBase &rewriter) {}

// TODO(zhiyuang): type check may block the conversion. Make sure we got all type infered!
/// LoadOp's Memory Slot Interaface
bool LoadOp::loadsFrom(const MemorySlot &slot) {
  return getRef() == slot.ptr;
}
bool LoadOp::storesTo(const MemorySlot &slot) { return false; }
Value LoadOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  llvm_unreachable("getStored should not be called on LoadOp");
}
// TODO(zhiyuang): check this. when do they use this call?
bool LoadOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, there will be enough
  // context to reconstruct the result of the load at removal time, so it can
  // be removed (provided it loads the exact stored value and is not
  // volatile).
  return blockingUse == slot.ptr && getRef() == slot.ptr;
         getResult().getType() == slot.elemType;
}
DeletionKind LoadOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition) {
  // `canUsesBeRemoved` checked this blocking use must be the loaded slot
  // pointer.
  rewriter.replaceAllUsesWith(getResult(), reachingDefinition);
  return DeletionKind::Delete;
}

/// StoreOp's Memory Slot Interaface
bool StoreOp::loadsFrom(const MemorySlot &slot) { return false; }
bool StoreOp::storesTo(const MemorySlot &slot) {
  return getOutput() == slot.ptr;
}
Value StoreOp::getStored(const MemorySlot &slot, RewriterBase &rewriter) {
  return getValue();
}
bool StoreOp::canUsesBeRemoved(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    SmallVectorImpl<OpOperand *> &newBlockingUses) {
  if (blockingUses.size() != 1)
    return false;
  Value blockingUse = (*blockingUses.begin())->get();
  // If the blocking use is the slot ptr itself, dropping the store is
  // fine, provided we are currently promoting its target value. Don't allow a
  // store OF the slot pointer, only INTO the slot pointer.
  return blockingUse == slot.ptr && getOutput() == slot.ptr;
         getValue() != slot.ptr && getValue().getType() == slot.elemType;
}
DeletionKind StoreOp::removeBlockingUses(
    const MemorySlot &slot, const SmallPtrSetImpl<OpOperand *> &blockingUses,
    RewriterBase &rewriter, Value reachingDefinition) {
  // `canUsesBeRemoved` checked this blocking use must be the stored slot
  // pointer.
  return DeletionKind::Delete;
}