// rope_style.h - how rotary position embedding (RoPE) pairs up the elements of one attention head.
#pragma once

namespace geneinfer {

//   Interleaved: (x[0], x[1]), (x[2], x[3]), ...        llama2.c
//   HalfSplit:   (x[i], x[i + head_dim/2])              Hugging Face Llama ("rotate_half")
enum class RopeStyle { Interleaved, HalfSplit };

// Frequency scaling of the "llama3" rope_scaling type (Llama 3.1 / 3.2), which stretches the slow rotations to a longer
// context: a rotation with wavelength > original_max_position / low_freq_factor gets its frequency divided by `factor`,
// one with wavelength < original_max_position / high_freq_factor keeps it, and in between the two are blended smoothly.
struct RopeScaling {
    bool enabled = false;
    double factor = 1.0;
    double low_freq_factor = 1.0;
    double high_freq_factor = 4.0;
    int original_max_position = 8192;
};

}  // namespace geneinfer
