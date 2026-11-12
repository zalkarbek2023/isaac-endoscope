#!/bin/bash
# Evaluate the autopilot with a given detector engine: 3 runs from different starts + 1 control run without polyps.
# Usage: ./eval_runs.sh [engine] [tag]     (default engine: out/polyp_sim.engine) -> out/eval_<tag>.txt
cd ~/robot-endo
ENG=${1:-$HOME/robot-endo/out/polyp_sim.engine}; TAG=${2:-sim}
[ -f out/detector.pid ] && kill $(cat out/detector.pid) 2>/dev/null; sleep 2
ENGINE=$ENG setsid nohup ~/medai-colon/.venv/bin/python detector_server.py > out/detector.log 2>&1 < /dev/null & echo $! > out/detector.pid
for i in $(seq 1 30); do grep -q "detector ready" out/detector.log && break; sleep 2; done
docker run --rm -u 0 -v ~/robot-endo/out:/o --entrypoint bash nvcr.io/nvidia/isaac-sim:6.2.0.1 -c "rm -rf /o/${TAG}_r* && chmod -R 777 /o" 2>/dev/null
for spec in "r1 0.000 0" "r2 0.010 1" "r3 0.020 2" "rc 0.000 3 control"; do
  set -- $spec
  ./run_isaac.sh endo_sim.py ${TAG}_$1 $2 $3 $4 > out/${TAG}_$1.log 2>&1
  grep SUMMARY out/${TAG}_$1.log | python3 -c "
import json,sys
for l in sys.stdin:
    s=json.loads(l.split('SUMMARY ',1)[1]); print(s['run'], 'control' if s['control'] else '', 'found', s['found_ids'], 'of', len(s['lesions']), '| alarms', s['alarms'], 'false', s['false_alarm_events'], '| false inspections', s['false_alarms'], '| s', s['seconds'], '| wall', s['wall_hits'])"
done | tee out/eval_${TAG}.txt
echo "EVAL DONE"
