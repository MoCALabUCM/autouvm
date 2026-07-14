#include "tools/uvm_advisor.h"
#include "utils/helper.h"
#include "utils/hash.h"
#include "gpu_patch.h"
#include "cpp_trace.h"
#include "py_frame.h"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <iostream>


using namespace yosemite;

#define SANITIZER_UVM_MEMORY_FLAG 0x6
#define LARGE_TENSOR_THRESHOLD 1048576

// AutoUVM offline analysis: max op gap (in scheduled-op order) still counted as
// short-term reuse for look-ahead affinity pinning.
#define AUTOUVM_AFFINITY_WINDOW 1

// Fallback prior (used only when no FLOP data is available): compute-bound
// operators tolerate migration latency, so the executor skips prefetching them.
static bool is_compute_bound_op(const std::string& name) {
    static const char* kCompute[] = {
        "matmul", "addmm", "addbmm", "baddbmm", "bmm", "::mm",
        "linear", "gemm", "gemv", "conv", "einsum", "scaled_dot_product"
    };
    for (auto p : kCompute) {
        if (name.find(p) != std::string::npos) return true;
    }
    return false;
}

// Three roofline categories (shared encoding with the online executor): a
// kernel is prefetched iff it is CGI-bound.
enum { ROOFLINE_UNKNOWN = -1, ROOFLINE_COMPUTE = 0, ROOFLINE_MEM = 1, ROOFLINE_CGI = 2 };

// Roofline roofs (compute GFLOP/s, GPU-mem and CGI GB/s); overridable via env.
// Defaults approximate an A100 80GB PCIe (FP32 peak, HBM2e BW, PCIe4 x16).
struct RooflineRoofs {
    double peak_gflops = 19500.0;   // AUTOUVM_PEAK_GFLOPS
    double mem_bw_gbps = 1935.0;    // AUTOUVM_MEM_BW_GBPS
    double cgi_bw_gbps = 25.0;      // AUTOUVM_CGI_BW_GBPS
    double margin      = 1.3;       // AUTOUVM_ROOF_MARGIN (slack for "at the roof")
};
static RooflineRoofs read_roofs() {
    RooflineRoofs r;
    if (const char* v = std::getenv("AUTOUVM_PEAK_GFLOPS")) r.peak_gflops = std::stod(v);
    if (const char* v = std::getenv("AUTOUVM_MEM_BW_GBPS")) r.mem_bw_gbps = std::stod(v);
    if (const char* v = std::getenv("AUTOUVM_CGI_BW_GBPS")) r.cgi_bw_gbps = std::stod(v);
    if (const char* v = std::getenv("AUTOUVM_ROOF_MARGIN")) r.margin      = std::stod(v);
    return r;
}

// Classify from FLOPs, migratable bytes, and (optional) measured time. time_s<=0
// means "assume resident" (bound by the memory/compute roof, never CGI).
static int roofline_classify(double flops, double bytes, double time_s,
                             const RooflineRoofs& r) {
    if (flops <= 0.0 || bytes <= 0.0) return ROOFLINE_UNKNOWN;
    double ai = flops / bytes;                          // FLOP per byte
    double cgi_roof = ai * r.cgi_bw_gbps;               // GFLOP/s at this AI
    double achieved = (time_s > 0.0) ? (flops / time_s / 1e9)
                                     : std::min(r.peak_gflops, ai * r.mem_bw_gbps);
    if (achieved <= cgi_roof * r.margin) return ROOFLINE_CGI;        // bound by CGI
    if (ai >= r.peak_gflops / r.mem_bw_gbps) return ROOFLINE_COMPUTE;// past mem ridge
    return ROOFLINE_MEM;
}

// Load a "value|kernel_name" metric file (roofline_flops.txt / roofline_time.txt),
// averaging duplicate launches of the same kernel name.
static std::unordered_map<std::string, double> load_kernel_metric(const std::string& path) {
    std::unordered_map<std::string, double> sum;
    std::unordered_map<std::string, uint64_t> cnt;
    std::ifstream in(path);
    std::string line;
    while (std::getline(in, line)) {
        size_t bar = line.find('|');
        if (bar == std::string::npos) continue;
        double val;
        try { val = std::stod(line.substr(0, bar)); } catch (...) { continue; }
        std::string name = line.substr(bar + 1);
        sum[name] += val;
        cnt[name] += 1;
    }
    std::unordered_map<std::string, double> avg;
    for (auto& kv : sum) avg[kv.first] = kv.second / (double) cnt[kv.first];
    return avg;
}

inline std::string vector2str(std::vector<std::string> &vec, int skip_first = 0, int skip_last = 0) {
    if (skip_first + skip_last > vec.size()) {
        printf("Skip first and skip last are larger than the vector size\n");
        return "";
    }
    std::string str;
    for (size_t i = skip_first; i < vec.size() - skip_last; i++) {
        str += vec[i] + "\n";
    }
    return str;
}


UVMAdvisor::UVMAdvisor() : Tool(UVM_ADVISOR) {
    init();
}


UVMAdvisor::~UVMAdvisor() {
}

void UVMAdvisor::init() {
    const char* env_name = std::getenv("ACCEL_PROF_HOME");
    std::string lib_path;
    if (env_name) {
        lib_path = std::string(env_name) + "/lib/libcompute_sanitizer.so";
    }
    init_backtrace(lib_path.c_str());

}


void UVMAdvisor::evt_callback(EventPtr_t evt) {
    switch (evt->evt_type) {
        case EventType_KERNEL_LAUNCH:
            kernel_start_callback(std::dynamic_pointer_cast<KernelLaunch_t>(evt));
            break;
        case EventType_KERNEL_END:
            kernel_end_callback(std::dynamic_pointer_cast<KernelEnd_t>(evt));
            break;
        case EventType_MEM_ALLOC:
            mem_alloc_callback(std::dynamic_pointer_cast<MemAlloc_t>(evt));
            break;
        case EventType_MEM_FREE:
            mem_free_callback(std::dynamic_pointer_cast<MemFree_t>(evt));
            break;
        case EventType_MEM_COPY:
            mem_cpy_callback(std::dynamic_pointer_cast<MemCpy_t>(evt));
            break;
        case EventType_MEM_SET:
            mem_set_callback(std::dynamic_pointer_cast<MemSet_t>(evt));
            break;
        case EventType_TEN_ALLOC:
            ten_alloc_callback(std::dynamic_pointer_cast<TenAlloc_t>(evt));
            break;
        case EventType_TEN_FREE:
            ten_free_callback(std::dynamic_pointer_cast<TenFree_t>(evt));
            break;
        case EventType_OP_START:
            op_start_callback(std::dynamic_pointer_cast<OpStart_t>(evt));
            break;
        case EventType_OP_END:
            op_end_callback(std::dynamic_pointer_cast<OpEnd_t>(evt));
            break;
        default:
            break;
    }
}


void UVMAdvisor::kernel_start_callback(std::shared_ptr<KernelLaunch_t> kernel) {
    opt_keys.kernel_id ++;
    kernel->key = opt_keys.kernel_id;
    kernel->timestamp = _timer.get();
    kernel_events.push_back(kernel);
    op_stats.pending_kernels++;
    _timer.increment(true);
}


void UVMAdvisor::kernel_end_callback(std::shared_ptr<KernelEnd_t> kernel) {
    _timer.increment(true);
}


void UVMAdvisor::mem_alloc_callback(std::shared_ptr<MemAlloc_t> mem) {
    mem_stats.current_mem_size += mem->size;
    mem_stats.max_mem_size = std::max(mem_stats.max_mem_size, mem_stats.current_mem_size);
    if (mem->alloc_type != SANITIZER_UVM_MEMORY_FLAG) {
        return;
    }

    opt_keys.mem_id ++;
    mem->key = opt_keys.mem_id;
    mem->timestamp = _timer.get();
    op_stats.pending_mem_alloc++;
    mem_stats.alloc_count++;
    mem_stats.alloc_size += mem->size;
    alloc_events.emplace(_timer.get(), mem);
    active_memories.emplace(mem->addr, mem);

    mem_alloc_during_this_op.insert(mem->addr);

    _timer.increment(true);
}


void UVMAdvisor::mem_free_callback(std::shared_ptr<MemFree_t> mem) {
    mem_stats.current_mem_size -= mem->size;
    if (mem->alloc_type != SANITIZER_UVM_MEMORY_FLAG) {
        return;
    }

    mem_stats.free_count++;
    mem_stats.free_size += mem->size;

    auto it = active_memories.find(mem->addr);
    assert(it != active_memories.end());
    active_memories.erase(it);

    _timer.increment(true);
}



void UVMAdvisor::mem_cpy_callback(std::shared_ptr<MemCpy_t> mem) {
    _timer.increment(true);
}


void UVMAdvisor::mem_set_callback(std::shared_ptr<MemSet_t> mem) {
    _timer.increment(true);
}

bool UVMAdvisor::find_uvm_tensor(uint64_t ptr) {
    for (auto mem : active_memories) {
        if (ptr >= mem.first && ptr < mem.first + mem.second->size) {
            return true;
        }
    }
    return false;
}


void UVMAdvisor::ten_alloc_callback(std::shared_ptr<TenAlloc_t> ten) {
    ten_stats.current_ten_size += ten->size;
    ten_stats.max_ten_size = std::max(ten_stats.max_ten_size, ten_stats.current_ten_size);
    if (ten->size <= LARGE_TENSOR_THRESHOLD) {
        return;
    }
    opt_keys.ten_id ++;

    if (!find_uvm_tensor(ten->addr)) {
        return;
    }

    
    ten->key = opt_keys.ten_id;
    op_stats.pending_ten_alloc++;
    ten_stats.alloc_count++;
    ten_stats.alloc_size += ten->size;

    ten->timestamp = _timer.get();
    tenalloc_events.emplace(_timer.get(), ten);
    active_tensors.emplace(ten->addr, ten);

    ten_alloc_during_this_op.insert(ten->addr);

    _timer.increment(true);
}


void UVMAdvisor::ten_free_callback(std::shared_ptr<TenFree_t> ten) {
    ten_stats.current_ten_size -= -ten->size;
    if (-ten->size <= LARGE_TENSOR_THRESHOLD) {
        return;
    }

    if (active_tensors.find(ten->addr) == active_tensors.end()) {
        return;
    }

    ten_stats.free_count++;
    ten_stats.free_size += -ten->size;

    auto it = active_tensors.find(ten->addr);
    assert(it != active_tensors.end());
    active_tensors.erase(it);

    _timer.increment(true);
}


void UVMAdvisor::op_start_callback(std::shared_ptr<OpStart_t> op) {
    opt_keys.op_id ++;
    op->key = opt_keys.op_id;
    op->timestamp = _timer.get();
    op_stack.push(op);
    op_stats.count++;
    op_stats.pending_ops++;

    _timer.increment(true);
}


void UVMAdvisor::op_end_callback(std::shared_ptr<OpEnd_t> op) {
    auto op_start = op_stack.top();
    op_stack.pop();
    if (op_stack.empty()) {
        if (op_stats.pending_kernels > 0 && kernel_resources.size() > 0) {
            assert(op_tables.find(op_start->timestamp) == op_tables.end());
            op_start->end_time = _timer.get();
            op_start->pending_kernels = op_stats.pending_kernels;
            op_start->pending_ops = op_stats.pending_ops;
            op_start->pending_mem_alloc = op_stats.pending_mem_alloc;
            op_start->pending_ten_alloc = op_stats.pending_ten_alloc;
            op_tables[op_start->timestamp] = std::make_pair(op_start, kernel_resources);
        }
        op_stats.group_count++;
        op_stats.pending_kernels = 0;
        op_stats.pending_ops = 0;
        op_stats.pending_mem_alloc = 0;
        op_stats.pending_ten_alloc = 0;
        kernel_resources.clear();
        ten_alloc_during_this_op.clear();
        mem_alloc_during_this_op.clear();
    }

    _timer.increment(true);
}

void UVMAdvisor::gpu_data_analysis(void* data, uint64_t size) {
    MemoryAccessTracker* tracker = (MemoryAccessTracker*)data;
    MemoryAccessState* states = tracker->access_state;
    TensorAccessState* tensor_states = tracker->tensor_access_state;

    MemAllocVec mem_alloc_vec;
    TenAllocVec ten_alloc_vec;

    for (uint32_t i = 0; i < states->size; i++) {
        if (states->touch[i] == 1) {
            auto mem = active_memories.find(states->start_end[i].start);
            // not allocated during this op
            if (mem_alloc_during_this_op.find(mem->second->addr) == mem_alloc_during_this_op.end()) {
                mem_alloc_vec.push_back(mem->second);
            }
        }
    }

    for (uint32_t i = 0; i < tensor_states->size; i++) {
        if (tensor_states->touch[i] == 1) {
            auto ten = active_tensors.find(tensor_states->start_end[i].start);
            // not allocated during this op
            if (ten_alloc_during_this_op.find(ten->second->addr) == ten_alloc_during_this_op.end()) {
                ten_alloc_vec.push_back(ten->second);
            }
        }
    }

    if (mem_alloc_vec.empty() && ten_alloc_vec.empty()) {
        return;
    }

    auto kernel = kernel_events.back();
    kernel_resources.push_back(std::make_tuple(kernel, mem_alloc_vec, ten_alloc_vec));
}


void UVMAdvisor::query_ranges(void* ranges, uint32_t limit, uint32_t* count) {
    MemoryRange* _ranges = (MemoryRange*)ranges;
    *count = 0;
    for (auto mem : active_memories) {
        _ranges[*count].start = mem.second->addr;
        _ranges[*count].end = mem.second->addr + mem.second->size;
        (*count)++;
        if (*count >= limit) {
            fprintf(stdout, "Warning: query_ranges limit reached\n");
            break;
        }
    }
}

void UVMAdvisor::query_tensors(void* ranges, uint32_t limit, uint32_t* count) {
    MemoryRange* _ranges = (MemoryRange*)ranges;
    *count = 0;
    for (auto ten : active_tensors) {
        _ranges[*count].start = ten.second->addr;
        _ranges[*count].end = ten.second->addr + ten.second->size;
        (*count)++;
        if (*count >= limit) {
            fprintf(stdout, "Warning: query_tensors limit reached\n");
            break;
        }
    }
}



void UVMAdvisor::print_callstack() {
    auto backtraces = get_backtrace();
    auto py_frames = get_pyframes();
    auto bt_str = vector2str(backtraces);
    auto pf_str = vector2str(py_frames);
    std::cout << bt_str << std::endl;
    std::cout << pf_str << std::endl;
}


void UVMAdvisor::flush() {
    FILE* out;
    std::string file_name = "uvm_advisor_opt.log";
    out = fopen(file_name.c_str(), "w");
    
    fprintf(out, "--------------------------------------------------------------------------------\n");
    fprintf(out, "%-12s max_size: %lu (%s)\n", 
            "[Memory]", mem_stats.max_mem_size, format_size(mem_stats.max_mem_size).c_str());
    fprintf(out, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[MemMalloc]", mem_stats.alloc_count, mem_stats.alloc_size, format_size(mem_stats.alloc_size).c_str());
    fprintf(out, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[MemFree]", mem_stats.free_count, mem_stats.free_size, format_size(mem_stats.free_size).c_str());

    fprintf(out, "%-12s max_size: %lu (%s)\n", 
            "[Tensor]", ten_stats.max_ten_size, format_size(ten_stats.max_ten_size).c_str());
    fprintf(out, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[TenMalloc]", ten_stats.alloc_count, ten_stats.alloc_size, format_size(ten_stats.alloc_size).c_str());
    fprintf(out, "%-12s count: %-10lu, size: %lu (%s)\n", 
            "[TenFree]", ten_stats.free_count, ten_stats.free_size, format_size(ten_stats.free_size).c_str());
    fprintf(out, "%-12s count: %-10lu\n", "[Op]", op_stats.count);
    fprintf(out, "%-12s count: %-10lu\n", "[OpGroup]", op_stats.group_count);
    fprintf(out, "--------------------------------------------------------------------------------\n");

    // ----------------------------------------------------------------------
    // AutoUVM offline analysis: look-ahead tensor affinity + roofline category.
    // These are written into the profile so the online executor can just read
    // them (like MemAlloc/TenAlloc) instead of recomputing at runtime. Per-op
    // FLOPs come from the roofline_flops/roofline_time profiling outputs, joined
    // by full kernel name (the trace truncates names, so we join here in memory).
    // ----------------------------------------------------------------------
    RooflineRoofs roofs = read_roofs();
    auto flops_map = load_kernel_metric("./out/roofline_flops.txt");
    auto time_map  = load_kernel_metric("./out/roofline_time.txt");   // milliseconds

    // 1) Per tensor: ordered op positions (affinity). Per op: FLOPs / time / bytes.
    std::vector<uint64_t> ordered_ops;
    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> ten_info;  // ten_id -> (addr, size)
    std::unordered_map<uint64_t, std::vector<int>> ten_positions;
    std::unordered_map<uint64_t, double> op_flops, op_time_ms, op_bytes;   // by op key
    int pos = 0;
    for (auto& it : op_tables) {
        auto op = it.second.first;
        ordered_ops.push_back(op->key);
        std::unordered_set<uint64_t> tens_this_op;
        double f = 0.0, t = 0.0;
        for (auto& kt : it.second.second) {
            auto kernel = std::get<0>(kt);
            auto fit = flops_map.find(kernel->kernel_name);
            if (fit != flops_map.end()) f += fit->second;
            auto tit = time_map.find(kernel->kernel_name);
            if (tit != time_map.end()) t += tit->second;
            for (auto& ten : std::get<2>(kt)) {
                tens_this_op.insert(ten->key);
                ten_info[ten->key] = std::make_pair(ten->addr, ten->size);
            }
        }
        double b = 0.0;
        for (uint64_t tid : tens_this_op) {
            b += ten_info[tid].second;
            ten_positions[tid].push_back(pos);
        }
        op_flops[op->key] = f;
        op_time_ms[op->key] = t;
        op_bytes[op->key] = b;
        ++pos;
    }
    // 2) A tensor reused across consecutive ops (gap <= window) forms a reuse
    //    run: pin it at the run's first op, release after the run's last op.
    std::unordered_map<uint64_t, std::vector<uint64_t>> pin_at, unpin_at;
    for (auto& kv : ten_positions) {
        auto& P = kv.second;   // increasing
        size_t i = 0;
        while (i < P.size()) {
            size_t j = i;
            while (j + 1 < P.size() && P[j + 1] - P[j] <= AUTOUVM_AFFINITY_WINDOW) ++j;
            if (j > i) {
                pin_at[ordered_ops[P[i]]].push_back(kv.first);
                unpin_at[ordered_ops[P[j]]].push_back(kv.first);
            }
            i = j + 1;
        }
    }

    fprintf(out, "================================================================================\n");
    for (auto& it : op_tables) {
        auto op = it.second.first;
        // Roofline category from FLOPs/bytes/time; fall back to the name prior
        // when FLOP data is unavailable. prefetch = 1 iff CGI-bound.
        double f = op_flops[op->key], b = op_bytes[op->key], t_ms = op_time_ms[op->key];
        int category = roofline_classify(f, b, t_ms / 1e3, roofs);
        int prefetch_hint;
        if (category == ROOFLINE_UNKNOWN) {
            prefetch_hint = is_compute_bound_op(op->op_name) ? 0 : 1;
        } else {
            prefetch_hint = (category == ROOFLINE_CGI) ? 1 : 0;
        }
        fprintf(out, "Op - %.30s, op_id: %lu, pending_ops: %lu, pending_kernels: %lu, pending_mem_alloc: %lu, pending_ten_alloc: %lu, flops: %.0f, bytes: %.0f, category: %d, prefetch: %d\n",
                op->op_name.c_str(), op->key, op->pending_ops, op->pending_kernels, op->pending_mem_alloc, op->pending_ten_alloc, f, b, category, prefetch_hint);
        // Affinity hints for this op (tensors to pin / release).
        auto pit = pin_at.find(op->key);
        if (pit != pin_at.end()) {
            fprintf(out, "       PinTen (%lu): ", pit->second.size());
            for (uint64_t tid : pit->second)
                fprintf(out, "%lu:(%lu, %lu), ", tid, ten_info[tid].first, ten_info[tid].second);
            fprintf(out, "\n");
        }
        auto uit = unpin_at.find(op->key);
        if (uit != unpin_at.end()) {
            fprintf(out, "       UnpinTen (%lu): ", uit->second.size());
            for (uint64_t tid : uit->second)
                fprintf(out, "%lu:(%lu, %lu), ", tid, ten_info[tid].first, ten_info[tid].second);
            fprintf(out, "\n");
        }
        for (auto& kernel_tuple : it.second.second) {
            auto kernel = std::get<0>(kernel_tuple);
            auto mem_alloc_vec = std::get<1>(kernel_tuple);
            auto ten_alloc_vec = std::get<2>(kernel_tuple);
            fprintf(out, "   Kernel: %.30s, kernel_id: %lu\n", kernel->kernel_name.c_str(), kernel->key);
            fprintf(out, "       MemAlloc (%lu): ", mem_alloc_vec.size());
            for (auto& mem : mem_alloc_vec) {
                fprintf(out, "%lu:(%lu, %lu), ", mem->key, mem->addr, mem->size);
            }
            fprintf(out, "\n");
            fprintf(out, "       TenAlloc (%lu): ", ten_alloc_vec.size());
            for (auto& ten : ten_alloc_vec) {
                fprintf(out, "%lu:(%lu, %lu), ", ten->key, ten->addr, ten->size);
            }
            fprintf(out, "\n");
        }
    }
    fclose(out);
}
