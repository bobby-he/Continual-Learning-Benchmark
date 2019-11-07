GPUID=$1
OUTDIR=outputs/split_MNIST_incremental_domain
REPEAT=1
mkdir -p $OUTDIR
CUDA_VISIBLE_DEVICES=0 python3 -u jl_iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --reg_coef 1 --damping 0.05 | tee ${OUTDIR}/jl_test_4.log

