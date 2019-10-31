GPUID=$1
OUTDIR=outputs/split_MNIST_incremental_domain
REPEAT=1
mkdir -p $OUTDIR

#python3 -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name EWC_online_mnist --lr 0.001 --reg_coef 600 750 850 900  | tee ${OUTDIR}/EWC_online_experimental.log
#python3 -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type customization  --agent_name EWC_mnist        --lr 0.001 --reg_coef 100 300 500 800  | tee ${OUTDIR}/EWC_experimental.log
python3 -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type experimental_regularization --agent_name SI  --lr 0.001 --reg_coef 2500 3000 3500     | tee ${OUTDIR}/SI_experimental2.log
#python3 -u iBatchLearn.py --gpuid $GPUID --repeat $REPEAT --optimizer Adam    --force_out_dim 2 --first_split_size 2 --other_split_size 2 --schedule 4 --batch_size 128 --model_name MLP400 --agent_type regularization --agent_name L2  --lr 0.001 --reg_coef 5      | tee ${OUTDIR}/L2_experimental.log

