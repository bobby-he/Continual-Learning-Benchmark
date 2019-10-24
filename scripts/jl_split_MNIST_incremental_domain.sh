GPUID=$1
OUTDIR=outputs/split_MNIST_incremental_domain
REPEAT=1
mkdir -p $OUTDIR
python3 -u jl_iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --reg_coef 25 50 60 70 80 90 100--damping 0.02 0.04 0.08 0.16 | tee ${OUTDIR}/jl_test_4.log

