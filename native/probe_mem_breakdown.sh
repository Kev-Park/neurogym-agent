#!/usr/bin/env bash
# Host-RAM breakdown of a running training, read-only. Run INSIDE the job's
# allocation:  srun --jobid=<JID> --overlap --ntasks=1 bash native/probe_mem_breakdown.sh
#
# Answers "why ~7.5G/env": splits the node's RSS/PSS by process ROLE (driver,
# ray:: env-runner, spawned fetch worker, Ray infra, MPS), counts fetch workers
# per runner (ppid), reports the job cgroup's memory.current (the number the
# OOM killer acts on; RSS double-counts shared pages, PSS does not), and dumps
# smaps_rollup for the largest runner + one fetch worker + the driver.
set -u
ME=$(id -un)
echo "== node $(hostname)  user $ME  $(date -u +%FT%TZ) =="
free -g | sed -n '1,2p'

echo "== job cgroup memory (what the OOM killer enforces) =="
CG=$(awk -F: '{print $3}' /proc/self/cgroup | head -1)
for f in /sys/fs/cgroup${CG}/memory.current /sys/fs/cgroup${CG}/memory.max \
         /sys/fs/cgroup/memory${CG}/memory.usage_in_bytes /sys/fs/cgroup/memory${CG}/memory.limit_in_bytes; do
  [ -r "$f" ] && awk -v f="$f" '{printf "%s = %.1f GB\n", f, $1/1073741824}' "$f"
done

# pid ppid rss(kB) args  -> add pss(kB) from smaps_rollup, classify by role.
TABLE=$(mktemp)
ps -u "$ME" -o pid=,ppid=,rss=,args= | grep -vE 'probe_mem_breakdown|^ *[0-9]+ +[0-9]+ +[0-9]+ +(ps |awk |grep |srun |slurmstepd)' \
| while read -r pid ppid rss args; do
    pss=$(awk '/^Pss:/{print $2}' "/proc/$pid/smaps_rollup" 2>/dev/null); pss=${pss:-0}
    case "$args" in
      *ngllib_agent.train*)                     role=driver ;;
      ray::*)                                   role=runner ;;
      *multiprocessing.spawn*|*spawn_main*)     role=fetch_worker ;;
      *resource_tracker*)                       role=mp_tracker ;;
      *raylet*|*gcs_server*|*dashboard*|*log_monitor*|*runtime_env*|*ray/autoscaler*|*ray::IDLE*) role=ray_infra ;;
      *nvidia-cuda-mps*)                        role=mps ;;
      *)                                        role=other ;;
    esac
    printf '%s %s %s %s %s %s\n' "$pid" "$ppid" "$rss" "$pss" "$role" "$(printf '%s' "$args" | cut -c1-70)"
  done > "$TABLE"

echo "== by role: n, RSS GB (double-counts shared), PSS GB (true share) =="
awk '{n[$5]++; r[$5]+=$3; p[$5]+=$4} END{for(k in n) printf "%-13s n=%3d  RSS=%6.1f GB  PSS=%6.1f GB  PSS/proc=%5.2f GB\n", k, n[k], r[k]/1048576, p[k]/1048576, p[k]/n[k]/1048576}' "$TABLE" | sort
awk '{r+=$3; p+=$4} END{printf "%-13s        RSS=%6.1f GB  PSS=%6.1f GB\n", "TOTAL", r/1048576, p/1048576}' "$TABLE"

echo "== fetch workers per runner (children of ray:: pids) =="
awk '$5=="runner"{run[$1]=1} $5=="fetch_worker"{c[$2]++} END{for(pid in run) printf "%d ", (pid in c)?c[pid]:0; printf "\n"}' "$TABLE" \
| tr ' ' '\n' | grep -v '^$' | sort -n | uniq -c | awk '{printf "  %s runners have %s fetch workers each\n", $1, $2}'
# workers whose parent is NOT a runner (e.g. grandchildren or driver-owned)
awk '$5=="runner"{run[$1]=1} $5=="fetch_worker" && !($2 in run){n++} END{printf "  fetch workers not parented by a runner: %d\n", n+0}' "$TABLE"

echo "== top 12 by RSS (GB) =="
sort -k3 -n -r "$TABLE" | head -12 | awk '{printf "%-13s pid=%-7s RSS=%5.2f PSS=%5.2f  %s\n", $5, $1, $3/1048576, $4/1048576, substr($0, index($0,$6))}'

dump() {  # role -> smaps_rollup of the largest pid in that role
  local pid; pid=$(awk -v r="$1" '$5==r' "$TABLE" | sort -k3 -n -r | head -1 | awk '{print $1}')
  [ -n "$pid" ] || { echo "== no $1 process =="; return; }
  echo "== smaps_rollup: largest $1 (pid $pid) =="
  grep -E '^(Rss|Pss|Shared_Clean|Shared_Dirty|Private_Clean|Private_Dirty|Anonymous|Swap):' "/proc/$pid/smaps_rollup" 2>/dev/null \
  | awk '{printf "  %-14s %7.2f GB\n", $1, $2/1048576}'
  # biggest individual mappings: anon heap vs CUDA/driver vs files
  echo "  top mappings (GB): "
  awk '/^[0-9a-f]+-[0-9a-f]+ /{name=$6; if(name=="") name="[anon]"} /^Rss:/{r[name]+=$2} END{for(k in r) printf "%12.2f  %s\n", r[k]/1048576, k}' "/proc/$pid/smaps" 2>/dev/null \
  | sort -n -r | head -6 | sed 's/^/    /'
}
dump runner
dump fetch_worker
dump driver
rm -f "$TABLE"
echo "MEMPROBE_DONE"
