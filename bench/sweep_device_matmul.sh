#!/bin/sh
# sweep_device_matmul.sh - times the tuning knobs of the device matmul (GI_DEV_PARTIALS x GI_DEV_TILE_ROWS).
#
#   sh bench/sweep_device_matmul.sh models/Llama-3.2-1B [cuda]
#
# For every combination: single-sequence decode speed (gi_generate) and the throughput of 8 and 32 concurrent
# requests (gi_serve_bench). Prints one line per combination.
MODEL="$1"
DEVICE="${2:-cuda}"
if [ -z "$MODEL" ]; then
    echo "usage: sh bench/sweep_device_matmul.sh MODEL_DIR [cuda|emu]" >&2
    exit 2
fi
echo "partials tile_rows | decode tok/s | 8 concurrent tok/s | 32 concurrent tok/s"
for P in 32 16 8; do
    for R in 4 8; do
        export GI_DEV_PARTIALS=$P GI_DEV_TILE_ROWS=$R
        DEC=$(./build/gi_generate --model "$MODEL" --device "$DEVICE" --weights native --max-seq-len 2048 --bos --temperature 0 \
              --prompt "The capital of France is" --steps 96 2>&1 | sed -n 's/.*decode \([0-9.]*\) tok\/s.*/\1/p')
        OUT=$(./build/gi_serve_bench --model "$MODEL" --device "$DEVICE" --weights native --concurrency 8,32 \
              --prompt-tokens 128 --decode-tokens 64 --runs 1 2>&1)
        C8=$(echo "$OUT" | awk '$1 == "8" {print $3}')
        C32=$(echo "$OUT" | awk '$1 == "32" {print $3}')
        SAME=$(echo "$OUT" | grep -c "identical to running alone")
        echo "$P $R | ${DEC:-failed} | ${C8:-failed} | ${C32:-failed} | levels identical to running alone: $SAME of 2"
    done
done
