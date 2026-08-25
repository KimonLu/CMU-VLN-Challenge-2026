#!/usr/bin/env bash
# Run one challenge question without RViz. The matching Unity scene must already
# be installed in iros2026_system before invoking this script.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <scene> <question-number> <run-name>" >&2
  exit 2
fi

scene=$1
question=$2
run_name=$3
headless_display=${HEADLESS_DISPLAY:-:99}
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
host_out="$repo_root/ai_module/tools/out/$run_name/$scene/q$question"
container_out="/home/docker/ai_module/tools/out/$run_name/$scene/q$question"

available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
swap_used_kib=$(awk '/SwapTotal:/ {total=$2} /SwapFree:/ {free=$2} END {print total-free}' /proc/meminfo)
c_free_kib=$(df -Pk /mnt/c | awk 'NR==2 {print $4}')
if (( available_kib < 1572864 || swap_used_kib > 3145728 || c_free_kib < 20971520 )); then
  echo "resource preflight failed: available=${available_kib}KiB swap_used=${swap_used_kib}KiB c_free=${c_free_kib}KiB" >&2
  exit 3
fi

mkdir -p "$host_out"

docker restart iros2026_system iros2026_ai_module >/dev/null

docker exec iros2026_system bash -lc "
  set -e
  base=/home/docker/autonomy_stack_mecanum_wheel_platform
  unity_parent=\"\$base/src/base_autonomy/vehicle_simulator/mesh/unity\"
  cached_scene=\"\$unity_parent/environment_$scene\"
  if [[ -d \"\$cached_scene\" ]]; then
    if [[ -L \"\$unity_parent/environment\" ]]; then
      rm \"\$unity_parent/environment\"
    elif [[ -e \"\$unity_parent/environment\" ]]; then
      echo \"refusing to replace non-symlink active environment\" >&2
      exit 4
    fi
    ln -s \"\$cached_scene\" \"\$unity_parent/environment\"
  fi
  test -x \"\$unity_parent/environment/Model.x86_64\"
  cd \"\$base\"
  rm -f /tmp/vln_unity.pid /tmp/vln_system.pid
  DISPLAY='$headless_display' nohup ./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 \
    > /tmp/vln_unity.log 2>&1 & echo \$! > /tmp/vln_unity.pid
  sleep 4
  source /opt/ros/jazzy/setup.bash
  source install/setup.bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI=file:///home/docker/cyclonedds.xml
  nohup ros2 launch vehicle_simulator system_simulation.launch \
    > /tmp/vln_system.log 2>&1 & echo \$! > /tmp/vln_system.pid
"
sleep 15

docker exec iros2026_ai_module bash -lc "
  set -e
  source /opt/ros/jazzy/setup.bash
  source /home/docker/ai_module/install/setup.bash
  mkdir -p '$container_out'
  nohup ros2 launch smart_vlm smart_vlm.launch.py \
    > '$container_out/smart_vlm.log' 2>&1 & echo \$! > '$container_out/smart_vlm.pid'
"
sleep 15

monitor_resources() {
  while true; do
    printf '%s\t' "$(date --iso-8601=seconds)"
    awk '/MemAvailable:/ {printf "mem_available_kib=%s\t", $2} /SwapTotal:/ {total=$2} /SwapFree:/ {free=$2} END {printf "swap_used_kib=%s\t", total-free}' /proc/meminfo
    df -Pk /mnt/c | awk 'NR==2 {printf "c_free_kib=%s\t", $4}'
    docker stats --no-stream --format '{{.Name}}={{.MemUsage}}' iros2026_system iros2026_ai_module | paste -sd ';' -
    sleep 10
  done
}
monitor_resources > "$host_out/resources.tsv" 2>&1 &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

set +e
timeout 660 docker exec iros2026_ai_module bash -lc "
  set -o pipefail
  source /opt/ros/jazzy/setup.bash
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
  export CYCLONEDDS_URI=file:///home/docker/cyclonedds.xml
  python3 /home/docker/ai_module/tools/run_one_question.py \
    --scene '$scene' --q '$question' --duration 600 \
    --questions-file /home/docker/ai_module/questions/questions.json \
    --out '$container_out' 2>&1 | tee '$container_out/evaluator.log'
"
eval_status=$?
set -e

kill "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
trap - EXIT

docker cp "iros2026_ai_module:$container_out/." "$host_out/" >/dev/null
docker cp iros2026_system:/tmp/vln_unity.log "$host_out/unity.log" >/dev/null
docker cp iros2026_system:/tmp/vln_system.log "$host_out/system.log" >/dev/null

if (( eval_status != 0 )); then
  echo "evaluator failed with status $eval_status; logs copied to $host_out" >&2
  exit "$eval_status"
fi
echo "completed $scene q$question -> $host_out"
