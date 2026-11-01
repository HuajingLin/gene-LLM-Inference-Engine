// rope_style.h - how rotary position embedding (RoPE) pairs up the elements of one attention head.
#pragma once

namespace geneinfer {

//   Interleaved: (x[0], x[1]), (x[2], x[3]), ...        llama2.c
//   HalfSplit:   (x[i], x[i + head_dim/2])              Hugging Face Llama ("rotate_half")
enum class RopeStyle { Interleaved, HalfSplit };

}  // namespace geneinfer
