#!/usr/bin/env bash
set -euo pipefail
mkdir -p logs
run_worker () {
  gpu="$1"; parity="$2"; i=0
  while IFS= read -r row; do
    if (( i % 2 == parity )); then
      name=$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["name"])' <<< "$row")
      echo "GPU $gpu: $name"
      CUDA_VISIBLE_DEVICES="$gpu" python simulate_hoomd_patchy_full.py <(printf '%s\n' "$row") > "logs/${name}_gpu${gpu}.log" 2>&1
    fi
    ((i+=1))
  done < <(python -c 'import json; [print(json.dumps(x)) for x in json.load(open("params.json"))]')
}
run_worker 0 0 &
p0=$!
run_worker 1 1 &
p1=$!
wait "$p0" "$p1"
echo "All runs complete."
