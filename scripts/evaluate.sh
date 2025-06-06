export PYTHONPATH=$PYTHONPATH:./


NFE=$1
GEN_SAMPLER=$2

SPLIT=train
PREFIX=e2h_ema_0.9999_420000_adapted/sample_420000
REF_PATH=assets/stats/edges2handbags_ref_64_data.npz
SAMPLE_NAME=samples_138567x64x64x3_nfe${NFE}.npz


if [[ $GEN_SAMPLER == "ddbm" ]]; then
    N=$(echo "$NFE" | awk '{print ($1 + 1) / 3}')
    N=$(printf "%.0f" "$N")
    SAMPLER="heun"
elif [[ $GEN_SAMPLER == "dpmsolver1" ]]; then
    N=$((NFE-1))
    ETA=0
    SAMPLER="dbim_eta=${ETA}"
elif [[ $GEN_SAMPLER == "dpmsolver2" ]]; then
    N=$((NFE-1))
    ORDER=2
    SAMPLER="dbim_order=${ORDER}"
elif [[ $GEN_SAMPLER == "dpmsolver3" ]]; then
    N=$((NFE-1))
    ORDER=3
    SAMPLER="dbim_order=${ORDER}"
fi


SAMPLE_DIR=workdir/${PREFIX}/split=${SPLIT}/${SAMPLER}/steps=${N}
SAMPLE_PATH=${SAMPLE_DIR}/${SAMPLE_NAME}


python evaluations/evaluator.py $REF_PATH $SAMPLE_PATH --metric fid
