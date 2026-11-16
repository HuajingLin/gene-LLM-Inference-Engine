#include "model/model.h"

#include <cstdlib>
#include <stdexcept>

#include "gpu/gpu_transformer.h"
#include "model/transformer.h"

namespace geneinfer {

std::unique_ptr<Model> make_model(const Checkpoint& checkpoint, const std::string& device, int max_seq_len, int n_threads,
                                  int max_batch) {
    std::string name = device;
    if (name.empty()) {
        const char* env = std::getenv("GI_DEVICE");
        name = (env && *env) ? env : "cpu";
    }
    if (name == "cpu") return std::make_unique<Transformer>(checkpoint, max_seq_len, n_threads, max_batch);
    return std::make_unique<GpuTransformer>(checkpoint, gpu::device_by_name(name), max_seq_len, max_batch);
}

}  // namespace geneinfer
