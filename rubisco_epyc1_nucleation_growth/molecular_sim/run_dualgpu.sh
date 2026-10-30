#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

while IFS= read -r row; do
  name=$(python -c 'import json,sys; print(json.loads(sys.stdin.read())["name"])' <<< "$row")
  echo "Running simulation: $name"

  python simulate_hoomd_patchy_full.py <(printf '%s\n' "$row") > "logs/${name}.log" 2>&1

  echo "Finished: $name"
done < <(python -c 'import json; [print(json.dumps(x)) for x in json.load(open("params.json"))]')

echo "All runs complete."