#include <ATen/record_function.h>
#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <sanitizer.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <memory>
#include <tuple>
#include <vector>
#include <map>
#include <regex>
#include <unordered_map>
#include <cassert>
#include <chrono>
#include <cxxabi.h>

#define CUDA_SAFECALL(call)                                      \
{                                                                \
    call;                                                        \
    cudaError err = cudaGetLastError();                          \
    if (cudaSuccess != err) {                                    \
        fprintf(                                                 \
            stderr,                                              \
            "[CUDA ERROR] '%s' in '%s:%i' - error: %s.\n",       \
            #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
        fflush(stderr);                                          \
        assert(false);                                           \
    }                                                            \
}


#define SANITIZER_UVM_MEMORY_FLAG 0x6
#define LARGE_TENSOR_THRESHOLD 1048576
#define FOREST_LINEAR_GRANULARITY 4194304

typedef enum {
    NO_PREFETCH = 0,
    OBJECT_LEVEL_PREFETCH = 1,
    TENSOR_LEVEL_PREFETCH = 2,
    FOREST_PREFETCH = 3,
    TENSOR_ROOFLINE_PREFETCH = 4,
    OBJECT_LEVEL_STATISTICS_COUNT = 5,
} PrefetchMode_t;
// ENV: PREFETCH_MODE
static PrefetchMode_t prefetch_mode = TENSOR_LEVEL_PREFETCH;


typedef struct {
    uint64_t op_id = 0;
    uint64_t kernel_id = 0;
    uint64_t mem_id = 0;
    uint64_t ten_id = 0;
} op_key_t;

static op_key_t op_key;
static std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> id_2_tensor_map;
static std::unordered_map<uint64_t, uint64_t> ptr_to_ten_id;
static std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> id_2_memory_map;

static std::vector<cudaStream_t> prefetch_streams;
static int num_prefetch_streams = 3;
static uint64_t stream_index = 0;

static int prefetch_device = 0;

static std::string profile_path = "uvm_advisor_opt.log";

static size_t global_prefetched_size = 0;
static size_t global_needed_size = 0;


using MemTenEntry = std::pair<std::vector<uint64_t>, std::vector<uint64_t>>;
std::unordered_map<int, MemTenEntry> prefetch_schedule;


static bool   autouvm_affinity = true;
static bool   autouvm_policy   = true;
static bool   autouvm_monitor  = true;
static int    autouvm_affinity_window = 1;
static uint64_t autouvm_monitor_warmup = 8;

enum {
    ROOFLINE_UNKNOWN = -1,
    ROOFLINE_COMPUTE = 0,
    ROOFLINE_MEM = 1,
    ROOFLINE_CGI = 2
};

static double autouvm_peak_gflops = 19500.0;
static double autouvm_mem_bw_gbps = 1935.0;
static double autouvm_cgi_bw_gbps = 25.0;
static double autouvm_roof_margin = 1.3;

static int roofline_classify(double flops, double bytes, double time_s) {
    if (flops <= 0.0 || bytes <= 0.0) return ROOFLINE_UNKNOWN;
    double ai = flops / bytes;
    double cgi_roof = ai * autouvm_cgi_bw_gbps;
    double achieved = (time_s > 0.0) ? (flops / time_s / 1e9)
                                     : std::min(autouvm_peak_gflops, ai * autouvm_mem_bw_gbps);
    if (achieved <= cgi_roof * autouvm_roof_margin) return ROOFLINE_CGI;
    if (ai >= autouvm_peak_gflops / autouvm_mem_bw_gbps) return ROOFLINE_COMPUTE;
    return ROOFLINE_MEM;
}

static std::vector<int> ordered_op_ids;

static std::unordered_map<int, std::vector<uint64_t>> affinity_pin_at;
static std::unordered_map<int, std::vector<uint64_t>> affinity_unpin_at;
static std::unordered_map<uint64_t, std::pair<void*, size_t>> pinned_tensor;

static std::unordered_map<int, double> op_flops;
static std::unordered_map<int, double> op_bytes_hint;
static std::unordered_map<int, int>    op_category;
static std::unordered_map<int, bool>   op_prefetch_hint;
static bool policy_hints_in_log = false;
static bool affinity_hints_in_log = false;

struct KernelStat {
    uint64_t count = 0;
    double sum_op_time_ms = 0.0;
    double sum_flops = 0.0;
    double sum_bytes = 0.0;
};
static std::unordered_map<std::string, KernelStat> kernel_stats;
// op_name -> category
static std::unordered_map<std::string, int> runtime_category;

static uint64_t roofline_prefetch_calls = 0;
static uint64_t autouvm_pin_calls = 0;
static uint64_t autouvm_skip_calls = 0;
static uint64_t autouvm_signature_count = 0;

static void parse_profile_output(const std::string& filename) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        return;
    }
    std::string line;

    int current_op_id = -1;

    auto extract_ids = [](const std::string& l) {
        std::vector<uint64_t> ids;
        std::regex entry_regex(R"((\d+):)");
        for (auto i = std::sregex_iterator(l.begin(), l.end(), entry_regex);
             i != std::sregex_iterator(); ++i) {
            ids.emplace_back((uint64_t) std::stoll((*i)[1].str()));
        }
        return ids;
    };

    while (std::getline(infile, line)) {
        if (line.rfind("Op -", 0) == 0) {
            size_t pos = line.find("op_id:");
            if (pos != std::string::npos) {
                current_op_id = std::stoi(line.substr(pos + 7));
                ordered_op_ids.push_back(current_op_id);
            }
            size_t fpos = line.find("flops:");
            if (fpos != std::string::npos) {
                op_flops[current_op_id] = std::stod(line.substr(fpos + 6));
            }
            size_t bpos = line.find("bytes:");
            if (bpos != std::string::npos) {
                op_bytes_hint[current_op_id] = std::stod(line.substr(bpos + 6));
            }
            size_t cpos = line.find("category:");
            if (cpos != std::string::npos) {
                op_category[current_op_id] = std::stoi(line.substr(cpos + 9));
                policy_hints_in_log = true;
            }
            size_t ppos = line.find("prefetch:");
            if (ppos != std::string::npos) {
                op_prefetch_hint[current_op_id] = (std::stoi(line.substr(ppos + 9)) != 0);
                policy_hints_in_log = true;
            }
        }

        else if (line.find("UnpinTen") != std::string::npos) {
            affinity_unpin_at[current_op_id] = extract_ids(line);
        }
        else if (line.find("PinTen") != std::string::npos) {
            affinity_pin_at[current_op_id] = extract_ids(line);
            affinity_hints_in_log = true;
        }

        // Parse MemAlloc
        else if (line.find("MemAlloc") != std::string::npos) {
            std::vector<uint64_t> mem_allocs;
            std::regex entry_regex(R"((\d+):)");
            auto begin = std::sregex_iterator(line.begin(), line.end(), entry_regex);
            auto end = std::sregex_iterator();
            for (auto i = begin; i != end; ++i) {
                std::smatch match = *i;
                uint64_t id = (uint64_t) std::stoi(match[1]);
                mem_allocs.emplace_back(id);
            }
            prefetch_schedule[current_op_id].first = mem_allocs;
        }

        // Parse TenAlloc
        else if (line.find("TenAlloc") != std::string::npos) {
            std::vector<uint64_t> ten_allocs;
            std::regex entry_regex(R"((\d+):)");
            auto begin = std::sregex_iterator(line.begin(), line.end(), entry_regex);
            auto end = std::sregex_iterator();
            for (auto i = begin; i != end; ++i) {
                std::smatch match = *i;
                uint64_t id = (uint64_t) std::stoi(match[1]);
                ten_allocs.emplace_back(id);
            }
            prefetch_schedule[current_op_id].second = ten_allocs;
        }
    }
}


static void no_prefetch(uint64_t op_id) {
}


static void tensor_level_prefetch(uint64_t op_id) {
    if (prefetch_schedule.find(op_id) == prefetch_schedule.end()) {
        return;
    }

    auto ten_allocs = prefetch_schedule[op_id].second;

    for (auto ten_alloc : ten_allocs) {
        auto tensor = id_2_tensor_map[ten_alloc];
        void* ptr = (void*) tensor.first;
        size_t size = tensor.second;

        CUDA_SAFECALL(
            cudaMemPrefetchAsync(ptr, size, prefetch_device, prefetch_streams[stream_index]));
        stream_index = (stream_index + 1) % num_prefetch_streams;
    }
}


static void object_level_prefetch(uint64_t op_id) {
    if (prefetch_schedule.find(op_id) == prefetch_schedule.end()) {
        return;
    }

    auto mem_allocs = prefetch_schedule[op_id].first;

    for (auto mem_alloc : mem_allocs) {
        auto memory = id_2_memory_map[mem_alloc];
        void* ptr = (void*) memory.first;
        size_t size = memory.second;

        CUDA_SAFECALL(
            cudaMemPrefetchAsync(ptr, size, prefetch_device, prefetch_streams[stream_index]));
        stream_index = (stream_index + 1) % num_prefetch_streams;
    }
}

static void forest_prefetch(uint64_t op_id) {
    if (prefetch_schedule.find(op_id) == prefetch_schedule.end()) {
        return;
    }

    auto mem_allocs = prefetch_schedule[op_id].first;

    for (auto mem_alloc : mem_allocs) {
        auto memory = id_2_memory_map[mem_alloc];
        void* ptr = (void*) memory.first;
        size_t size = memory.second;

        size_t start_ptr = (size_t)ptr;
        if (size <= FOREST_LINEAR_GRANULARITY) {
            CUDA_SAFECALL(
                cudaMemPrefetchAsync(
                    (void*)start_ptr, size, prefetch_device, prefetch_streams[stream_index]));
        } else {
            CUDA_SAFECALL(
                cudaMemPrefetchAsync(
                    (void*)start_ptr,
                    FOREST_LINEAR_GRANULARITY,
                    prefetch_device,
                    prefetch_streams[stream_index]));
            start_ptr += FOREST_LINEAR_GRANULARITY;
            size -= FOREST_LINEAR_GRANULARITY;
        }
        stream_index = (stream_index + 1) % num_prefetch_streams;
    }
}


static void compute_affinity() {
    std::unordered_map<uint64_t, std::vector<int>> ten_to_positions;
    for (size_t pos = 0; pos < ordered_op_ids.size(); ++pos) {
        auto it = prefetch_schedule.find(ordered_op_ids[pos]);
        if (it == prefetch_schedule.end()) continue;
        for (uint64_t ten_id : it->second.second) {
            ten_to_positions[ten_id].push_back((int)pos);
        }
    }
    for (auto& kv : ten_to_positions) {
        auto& positions = kv.second;
        size_t i = 0;
        while (i < positions.size()) {
            size_t j = i;
            while (j + 1 < positions.size() &&
                   positions[j + 1] - positions[j] <= autouvm_affinity_window) {
                ++j;
            }
            if (j > i) {
                affinity_pin_at[ordered_op_ids[positions[i]]].push_back(kv.first);
                affinity_unpin_at[ordered_op_ids[positions[j]]].push_back(kv.first);
            }
            i = j + 1;
        }
    }
}

static void affinity_pin(uint64_t ten_id) {
    auto it = id_2_tensor_map.find(ten_id);
    if (it == id_2_tensor_map.end()) return;
    void* ptr = (void*) it->second.first;
    size_t size = it->second.second;
    cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, prefetch_device);
    cudaGetLastError();
    pinned_tensor[ten_id] = std::make_pair(ptr, size);
    autouvm_pin_calls++;
}

static void affinity_unpin(uint64_t ten_id) {
    auto it = pinned_tensor.find(ten_id);
    if (it == pinned_tensor.end()) return;
    cudaMemAdvise(it->second.first, it->second.second,
                  cudaMemAdviseUnsetPreferredLocation, prefetch_device);
    cudaGetLastError();
    pinned_tensor.erase(it);
}

static bool is_compute_bound_name(const std::string& name) {
    static const char* kCompute[] = {
        "matmul", "addmm", "addbmm", "baddbmm", "bmm", "::mm",
        "linear", "gemm", "gemv", "conv", "einsum", "scaled_dot_product"
    };
    for (auto p : kCompute) {
        if (name.find(p) != std::string::npos) return true;
    }
    return false;
}

static bool roofline_should_prefetch(uint64_t op_id, const std::string& op_name) {
    if (!autouvm_policy) return true;

    if (autouvm_monitor) {
        auto rit = runtime_category.find(op_name);
        if (rit != runtime_category.end() && rit->second != ROOFLINE_UNKNOWN) {
            return rit->second == ROOFLINE_CGI;
        }
    }
    auto cit = op_category.find((int)op_id);
    if (cit != op_category.end() && cit->second != ROOFLINE_UNKNOWN) {
        return cit->second == ROOFLINE_CGI;
    }
    auto hint = op_prefetch_hint.find((int)op_id);
    if (hint != op_prefetch_hint.end()) {
        return hint->second;
    }
    return !is_compute_bound_name(op_name);
}

static void monitor_update(const std::string& op_name, uint64_t op_id, double op_time_ms) {
    auto it = kernel_stats.find(op_name);
    if (it == kernel_stats.end()) {
        it = kernel_stats.emplace(op_name, KernelStat{}).first;
        autouvm_signature_count++;
    }
    auto& st = it->second;
    st.count++;
    st.sum_op_time_ms += op_time_ms;
    auto fit = op_flops.find((int)op_id);
    auto bit = op_bytes_hint.find((int)op_id);
    if (fit != op_flops.end()) st.sum_flops += fit->second;
    if (bit != op_bytes_hint.end()) st.sum_bytes += bit->second;

    if (st.count >= autouvm_monitor_warmup) {
        double avg_t_s  = (st.sum_op_time_ms / st.count) / 1e3;
        double avg_flop = st.sum_flops / st.count;
        double avg_byte = st.sum_bytes / st.count;
        int c = roofline_classify(avg_flop, avg_byte, avg_t_s);
        if (c != ROOFLINE_UNKNOWN) runtime_category[op_name] = c;
    }
}

static void roofline_prefetch(uint64_t op_id, const std::string& op_name) {
    auto sched = prefetch_schedule.find((int)op_id);
    if (sched == prefetch_schedule.end()) {
        return;
    }
    auto& ten_allocs = sched->second.second;

    if (autouvm_affinity) {
        auto pit = affinity_pin_at.find((int)op_id);
        if (pit != affinity_pin_at.end()) {
            for (uint64_t ten_id : pit->second) affinity_pin(ten_id);
        }
    }

    if (roofline_should_prefetch(op_id, op_name)) {
        for (uint64_t ten_id : ten_allocs) {
            auto t = id_2_tensor_map.find(ten_id);
            if (t == id_2_tensor_map.end()) continue;
            void* ptr = (void*) t->second.first;
            size_t size = t->second.second;
            CUDA_SAFECALL(
                cudaMemPrefetchAsync(ptr, size, prefetch_device, prefetch_streams[stream_index]));
            stream_index = (stream_index + 1) % num_prefetch_streams;
            roofline_prefetch_calls++;
        }
    } else {
        autouvm_skip_calls++;
    }
}


static void object_level_statistics_count(uint64_t op_id) {
    if (prefetch_schedule.find(op_id) == prefetch_schedule.end()) {
        return;
    }

    auto mem_allocs = prefetch_schedule[op_id].first;

    size_t prefetched_size = 0;
    for (auto mem_alloc : mem_allocs) {
        auto memory = id_2_memory_map[mem_alloc];
        void* ptr = (void*) memory.first;
        size_t size = memory.second;

        prefetched_size += size;
        CUDA_SAFECALL(
            cudaMemPrefetchAsync(ptr, size, prefetch_device, prefetch_streams[stream_index]));
        stream_index = (stream_index + 1) % num_prefetch_streams;
    }
    global_prefetched_size += prefetched_size;

    auto ten_allocs = prefetch_schedule[op_id].second;
    for (auto ten_alloc : ten_allocs) {
        auto tensor = id_2_tensor_map[ten_alloc];
        size_t size = tensor.second;
        global_needed_size += size;
    }
}


////////////////////////////////////////////////////////////////////////////////
struct OperatorCallbackContext : at::ObserverContext {
    std::string op_name;
    uint64_t op_id = 0;
    std::chrono::steady_clock::time_point start;
};

static void operator_start(const at::RecordFunction& fn, at::ObserverContext* ctx) {
    op_key.op_id++;
    if (prefetch_mode == TENSOR_LEVEL_PREFETCH) {
        tensor_level_prefetch(op_key.op_id);
    } else if (prefetch_mode == OBJECT_LEVEL_PREFETCH) {
        object_level_prefetch(op_key.op_id);
    } else if (prefetch_mode == NO_PREFETCH) {
        no_prefetch(op_key.op_id);
    } else if (prefetch_mode == FOREST_PREFETCH) {
        forest_prefetch(op_key.op_id);
    } else if (prefetch_mode == TENSOR_ROOFLINE_PREFETCH) {
        auto* c = static_cast<OperatorCallbackContext*>(ctx);
        c->op_id = op_key.op_id;
        c->op_name = fn.name();
        roofline_prefetch(op_key.op_id, c->op_name);
        c->start = std::chrono::steady_clock::now();
    } else if (prefetch_mode == OBJECT_LEVEL_STATISTICS_COUNT) {
        object_level_statistics_count(op_key.op_id);
    }
}


static void operator_end(const at::RecordFunction& fn, at::ObserverContext* ctx) {
    if (prefetch_mode != TENSOR_ROOFLINE_PREFETCH) {
        return;
    }
    auto* c = static_cast<OperatorCallbackContext*>(ctx);
    if (autouvm_monitor &&
        prefetch_schedule.find((int)c->op_id) != prefetch_schedule.end()) {
        double op_time_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - c->start).count();
        monitor_update(c->op_name, c->op_id, op_time_ms);
    }
    if (autouvm_affinity) {
        auto uit = affinity_unpin_at.find((int)c->op_id);
        if (uit != affinity_unpin_at.end()) {
            for (uint64_t ten_id : uit->second) affinity_unpin(ten_id);
        }
    }
}


static void tensor_malloc(void* ptr, int64_t alloc_size, int64_t total_allocated,
                           int64_t total_reserved, c10::Device device) {
    if (alloc_size <= LARGE_TENSOR_THRESHOLD) {
        return;
    }
    op_key.ten_id++;
    id_2_tensor_map[op_key.ten_id] = std::make_pair((uint64_t)ptr, alloc_size);
    ptr_to_ten_id[(uint64_t)ptr] = op_key.ten_id;
}

static void tensor_free(void* ptr, int64_t alloc_size, int64_t total_allocated,
                           int64_t total_reserved, c10::Device device) {
    if ((-alloc_size) <= LARGE_TENSOR_THRESHOLD) {
        return;
    }
    auto ten_id = ptr_to_ten_id[(uint64_t)ptr];
    id_2_tensor_map.erase(ten_id);
    ptr_to_ten_id.erase((uint64_t)ptr);
}

class OperatorCallback {
public:
    static OperatorCallback& getInstance() {
        static OperatorCallback instance;
        return instance;
    }

private:
    OperatorCallback() {
        auto callback = at::RecordFunctionCallback(
            &OperatorCallback::startCallbackStatic,
            &OperatorCallback::endCallbackStatic
        ).scopes({at::RecordScope::FUNCTION});

        handle_ = at::addGlobalCallback(callback);
    }

    static std::unique_ptr<at::ObserverContext> startCallbackStatic(
        const at::RecordFunction& fn) {
        return getInstance().onStart(fn);
    }

    static void endCallbackStatic(const at::RecordFunction& fn,
        at::ObserverContext* ctx) {
        getInstance().onEnd(fn, ctx);
    }

    std::unique_ptr<at::ObserverContext> onStart(const at::RecordFunction& fn) {
        auto ctx = std::make_unique<OperatorCallbackContext>();
        operator_start(fn, ctx.get());
        return ctx;
    }

    void onEnd(const at::RecordFunction& fn, at::ObserverContext* ctx) {
        operator_end(fn, ctx);
    }

    at::CallbackHandle handle_;
};


class TensorCallback{
public:
    static TensorCallback& getInstance() {
        static TensorCallback instance;
        if (!instance.is_enabled) {
            instance.is_enabled = true;
            auto profiler = instance.new_memory_reporting_info();
            c10::ThreadLocalDebugInfo::_push(
                c10::DebugInfoKind::PROFILER_STATE, profiler);
        }
        return instance;
    }

private:
    TensorCallback() = default;
    ~TensorCallback() = default;

    class MemoryReportingInfo : public c10::MemoryReportingInfoBase {
    public:
        MemoryReportingInfo() = default;
        bool memoryProfilingEnabled() const override {
            return true;
        }

        #if TORCH_VERSION_MAJOR >= 2
        void reportMemoryUsage(void* ptr, int64_t alloc_size, size_t total_allocated,
                            size_t total_reserved, c10::Device device) override {
            if (device.is_cuda()) {
                if (alloc_size > 0) {
                    TensorCallback::getInstance().onMalloc(
                        ptr, alloc_size, total_allocated, total_reserved, device);
                } else {
                    TensorCallback::getInstance().onFree(
                        ptr, alloc_size, total_allocated, total_reserved, device);
                }
            }
        }
        #else
        void reportMemoryUsage(void* ptr, int64_t alloc_size, int64_t total_allocated,
                            int64_t total_reserved, c10::Device device) override {
            if (device.is_cuda()) {
                if (alloc_size > 0) {
                    TensorCallback::getInstance().onMalloc(
                        ptr, alloc_size, total_allocated, total_reserved, device);
                } else {
                    TensorCallback::getInstance().onFree(
                        ptr, alloc_size, total_allocated, total_reserved, device);
                }
            }
        }
        #endif
    };

    bool is_enabled = false;

    std::shared_ptr<MemoryReportingInfo> new_memory_reporting_info() {
        return std::make_shared<MemoryReportingInfo>();
    }

    void onMalloc(void* ptr, int64_t alloc_size, int64_t total_allocated,
                           int64_t total_reserved, c10::Device device) {
        tensor_malloc(ptr, alloc_size, total_allocated, total_reserved, device);
    }

    void onFree(void* ptr, int64_t alloc_size, int64_t total_allocated,
                           int64_t total_reserved, c10::Device device) {
        tensor_free(ptr, alloc_size, total_allocated, total_reserved, device);
    }
};

static const char* get_demangled_name(const char* function) {
    /// demangle function name
    const char *func_name = function;
    int status;
    char *demangled = abi::__cxa_demangle(function, nullptr, nullptr, &status);
    if (status == 0) {
        func_name = demangled;
    }

    return func_name;
}

static int init_prefetch_streams() {
    for (int i = 0; i < num_prefetch_streams; i++) {
        cudaStream_t prefetch_stream = nullptr;
        CUDA_SAFECALL(
            cudaStreamCreateWithFlags(&prefetch_stream, cudaStreamNonBlocking));
        prefetch_streams.push_back(prefetch_stream);
    }

    return 0;
}

void cs_callback(
    void* userdata,
    Sanitizer_CallbackDomain domain,
    Sanitizer_CallbackId cbid,
    const void* cbdata)
{
    if (domain == SANITIZER_CB_DOMAIN_RESOURCE) {
        if (cbid == SANITIZER_CBID_RESOURCE_DEVICE_MEMORY_ALLOC) {
            if (prefetch_mode == TENSOR_LEVEL_PREFETCH
                || prefetch_mode == TENSOR_ROOFLINE_PREFETCH) {
                return;
            }

            auto *pModuleData = (Sanitizer_ResourceMemoryData *)cbdata;
            if (pModuleData->flags != SANITIZER_UVM_MEMORY_FLAG) {
                return;
            }
            op_key.mem_id++;
            id_2_memory_map[op_key.mem_id] = std::make_pair(
                        (uint64_t)pModuleData->address, pModuleData->size);
        }
    }
}

static int compute_sanitizer() {
    Sanitizer_SubscriberHandle handle;
    sanitizerSubscribe(&handle, cs_callback, nullptr);
    sanitizerEnableDomain(1, handle, SANITIZER_CB_DOMAIN_RESOURCE);
    return 0;
}


static volatile int debug_flag = 0;
static int init_all_instances() {
    const char* debug_flag_str = std::getenv("DEBUG_FLAG");
    if (debug_flag_str) {
        debug_flag = std::stoi(debug_flag_str);
    }
    while (debug_flag) {}

    const char* prefetch_mode_str = std::getenv("PREFETCH_MODE");
    if (prefetch_mode_str) {
        prefetch_mode = static_cast<PrefetchMode_t>(std::stoi(prefetch_mode_str));
    }

    const char* prefetch_device_str = std::getenv("PREFETCH_DEVICE");
    if (prefetch_device_str) {
        prefetch_device = std::stoi(prefetch_device_str);
    }
    const char* profile_path_str = std::getenv("PREFETCH_PROFILE");
    if (profile_path_str) {
        profile_path = profile_path_str;
    }
    if (prefetch_mode == NO_PREFETCH) {
        printf("PREFETCH_MODE: NO PREFETCH\n");
    } else if (prefetch_mode == OBJECT_LEVEL_PREFETCH) {
        printf("PREFETCH_MODE: OBJECT LEVEL PREFETCH\n");
    } else if (prefetch_mode == TENSOR_LEVEL_PREFETCH) {
        printf("PREFETCH_MODE: TENSOR LEVEL PREFETCH\n");
    } else if (prefetch_mode == FOREST_PREFETCH) {
        printf("PREFETCH_MODE: FOREST PREFETCH\n");
    } else if (prefetch_mode == TENSOR_ROOFLINE_PREFETCH) {
        printf("PREFETCH_MODE: TENSOR ROOFLINE PREFETCH (AutoUVM)\n");
    } else if (prefetch_mode == OBJECT_LEVEL_STATISTICS_COUNT) {
        printf("PREFETCH_MODE: OBJECT LEVEL STATISTICS COUNT\n");
    }

    auto env_bool = [](const char* k, bool def) {
        const char* v = std::getenv(k);
        return v ? (std::stoi(v) != 0) : def;
    };
    if (prefetch_mode == TENSOR_ROOFLINE_PREFETCH) {
        autouvm_affinity = env_bool("AUTOUVM_AFFINITY", autouvm_affinity);
        autouvm_policy   = env_bool("AUTOUVM_POLICY",   autouvm_policy);
        autouvm_monitor  = env_bool("AUTOUVM_MONITOR",  autouvm_monitor);
        if (const char* v = std::getenv("AUTOUVM_AFFINITY_WINDOW")) {
            autouvm_affinity_window = std::stoi(v);
        }
        if (const char* v = std::getenv("AUTOUVM_MONITOR_WARMUP")) {
            autouvm_monitor_warmup = std::stoull(v);
        }
        if (const char* v = std::getenv("AUTOUVM_PEAK_GFLOPS")) {
            autouvm_peak_gflops = std::stod(v);
        }
        if (const char* v = std::getenv("AUTOUVM_MEM_BW_GBPS")) {
            autouvm_mem_bw_gbps = std::stod(v);
        }
        if (const char* v = std::getenv("AUTOUVM_CGI_BW_GBPS")) {
            autouvm_cgi_bw_gbps = std::stod(v);
        }
        if (const char* v = std::getenv("AUTOUVM_ROOF_MARGIN")) {
            autouvm_roof_margin = std::stod(v);
        }
    }

    parse_profile_output(profile_path);
    if (prefetch_mode == TENSOR_ROOFLINE_PREFETCH) {
        if (autouvm_affinity && !affinity_hints_in_log) {
            compute_affinity();
        }
        size_t pinned = 0;
        for (auto& kv : affinity_pin_at) pinned += kv.second.size();
    }
    static auto& operator_instance = OperatorCallback::getInstance();
    static auto& tensor_instance = TensorCallback::getInstance();
    static auto init_sanitizer = compute_sanitizer();
    static auto init_streams = init_prefetch_streams();
    return 0;
}

void cleanup(void) {
    if (prefetch_mode == TENSOR_ROOFLINE_PREFETCH) {
        return;
    }
    printf("global_prefetched_size:  %.3f MiB (%lu bytes, %.3f GB)\n",
        (float)global_prefetched_size / (1024 * 1024),
        global_prefetched_size,
        (float)global_prefetched_size / (1024 * 1024 * 1024));
    printf("global_needed_size:      %.3f MiB (%lu bytes, %.3f GB)\n",
        (float)global_needed_size / (1024 * 1024),
        global_needed_size,
        (float)global_needed_size / (1024 * 1024 * 1024));
    fflush(stdout);
}

__attribute__((constructor))
void initializer(void) {
    atexit(cleanup);
}

static int global_init_instance = init_all_instances();
