# Source this on Great Lakes or Whistler: `source setup/paths.sh`
export TURBO=/nfs/turbo/coe-chaijy-unreplicated/dannyyu
export REPLICA_DIR=$TURBO/datasets/replica_cad
export PP_DATA=$TURBO/datasets/perspective_pilot
export PP_RESULTS=$TURBO/results/perspective_pilot
export HF_HOME=$TURBO/models
export CONDA_ENVS=$TURBO/envs

mkdir -p "$REPLICA_DIR" "$PP_DATA" "$PP_RESULTS" "$HF_HOME" "$CONDA_ENVS"
