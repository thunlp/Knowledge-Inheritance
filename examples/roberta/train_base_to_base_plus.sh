TOTAL_UPDATES=125000    # Total number of training steps
TOTAL_UPDATES_DISTIL=55000    # Total number of distillation training steps
WARMUP_UPDATES=11000    # Warmup the learning rate over this many updates
PEAK_LR=0.00035        # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SENTENCES=16        # Number of sequences per batch (batch size)
UPDATE_FREQ=16          # Increase the batch size 16x
arch=roberta_base_plus
arch_distil_from=roberta_base
restore_file_distil_from=***your-teacher-model-path***
restore_file_checkpoint_distil_from=checkpoint_last.pt
logdir=log_base_to_base_plus
save_dir=checkpoint_base_to_base_plus
DATA_DIR=data-bin/corpus_all

python ../../fairseq_cli/train.py --fp16 $DATA_DIR \
        --task back_distil --criterion masked_lm_distil \
	    --arch $arch --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
		--optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
		--lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
		--dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
		--batch-size $MAX_SENTENCES --update-freq $UPDATE_FREQ \
		--max-update $TOTAL_UPDATES --log-format json --log-interval 100 \
		--max-update-distil $TOTAL_UPDATES_DISTIL \
		--tensorboard-logdir $logdir \
		--skip-invalid-size-inputs-valid-test \
		--save-dir $save_dir \
		--fixed-validation-seed 0 \
		--ddp-backend no_c10d \
		--arch_distil_from $arch_distil_from \
		--restore-file-distil-from $restore_file_distil_from \
		--restore-file-checkpoint-distil-from $restore_file_checkpoint_distil_from \
		--temperature_distil 2 \
		--restrict_ce_to_mask \
		--save-interval-updates 2500
