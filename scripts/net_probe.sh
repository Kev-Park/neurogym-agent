#!/bin/bash
# Reachability probe: can THIS node reach <host> on <port>? Used to diagnose whether
# a cross-node ray failure is routing (ping FAIL), a port/firewall (ping OK, tcp
# BLOCKED), or deeper (both OK but ray still fails). Runs natively on a compute node.
#   bash scripts/net_probe.sh <host> <port>
H="$1"; P="$2"
echo -n "[netprobe] $(hostname) -> $H ping: "
ping -c1 -W2 "$H" >/dev/null 2>&1 && echo OK || echo FAIL
echo -n "[netprobe] $(hostname) -> $H:$P tcp: "
timeout 5 bash -c "cat </dev/null >/dev/tcp/$H/$P" 2>/dev/null && echo OPEN || echo BLOCKED
