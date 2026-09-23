#pragma once

#include "runtime/logical_kv_store.h"

namespace ninfer::targets::qwen3_6::detail {

using ::ninfer::runtime::HostKVExtentCapability;
using ::ninfer::runtime::HostKVPageReplica;
using ::ninfer::runtime::LogicalKVPageHandle;
using ::ninfer::runtime::KVAddressSpaceHandle;
using ::ninfer::runtime::KVActivationReservation;
using ::ninfer::runtime::KVPrefixForkReservation;
using ::ninfer::runtime::KVActiveSnapshotShape;
using ::ninfer::runtime::KVActiveSnapshotReservation;
using ::ninfer::runtime::LogicalKVPageStore;
using ::ninfer::runtime::KVAddressSpaceStore;

} // namespace ninfer::targets::qwen3_6::detail
