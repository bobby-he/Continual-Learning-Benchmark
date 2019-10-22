GPUID=$1
OUTDIR=outputs/split_MNIST_incremental_domain
REPEAT=1
mkdir -p $OUTDIR
python3 -u jl_iBatchLearn.py --gpuid $GPUID --repeat $REPEAT  | tee ${OUTDIR}/jl.log

