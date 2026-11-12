#!/bin/bash
# synthetic data (Isaac) -> fine-tune (host venv) -> TensorRT engine. Detached; log: out/sim_train.log
cd ~/robot-endo
docker run --rm -u 0 -v ~/robot-endo/out:/o --entrypoint bash nvcr.io/nvidia/isaac-sim:6.2.0.1 -c "rm -rf /o/synth && chmod -R 777 /o"
echo "$(date +%T) synth start"
./run_isaac.sh synth_data.py ${1:-1500} 1 > out/synth.log 2>&1
grep -E "SYNTH DONE|Traceback" out/synth.log
echo "$(date +%T) finetune start"
~/medai-colon/.venv/bin/python finetune.py 2>&1 | grep -vE "^\s*$" | grep -E "train |REAL-VAL|FINETUNE|Error|Traceback|epoch|Epoch" | tail -40
echo "$(date +%T) ALL DONE"
