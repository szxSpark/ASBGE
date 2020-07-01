#!/bin/bash
set -x


SAVEPATH=/home/aistudio/work/lcsts/models/seass
DATAHOME=/home/aistudio/work/lcsts
# dev_ref 要用 char

python train.py \
       -save_path $SAVEPATH \
       -log_home $SAVEPATH \
       -online_process_data \
       -train_src ${DATAHOME}/train/subword/train.article.txt -src_vocab ${DATAHOME}/train/subword/source.vocab \
       -train_tgt ${DATAHOME}/train/subword/train.title.txt -tgt_vocab ${DATAHOME}/train/subword/target.vocab \
       -dev_input_src ${DATAHOME}/dev/subword/valid.article.txt -dev_ref ${DATAHOME}/dev/valid.title.char.txt \
       -layers 1 -enc_rnn_size 512 -brnn -word_vec_size 512 -dropout 0.5 \
       -batch_size 32 -beam_size 1 \
       -epochs 200 \
       -optim adam -learning_rate 0.001 \
       -gpus 0 \
       -curriculum 0 -extra_shuffle \
       -start_eval_batch 160000 -eval_per_batch 40000 \
       -log_interval 5000 
       -seed 12345 -cuda_seed 12345 \
       -max_sent_length 30 \
       -max_sent_length_source 100 \
       -subword True -is_save True \
       -pointer_gen False \
       -is_coverage False  -cov_loss_wt 1.0 \
       -share_embedding False \
       -halve_lr_bad_count 6

    #   -use_sentence_bucket_embedding False -sentence_embedding_size 150 -bucket 2 -max_bucket_num 100 \
    #   -lm_model_file /home/aistudio/work/toutiao_word/lm_pretrain_para_lstm_layer1_bptt30_tied.pkl
    # -checkpoint_file /home/aistudio/work/toutiao_word/models/seass/model_e89.pt




#cat ../train.article.txt ../train.title.txt | subword-nmt learn-bpe -s 20000 -o codes
#subword-nmt apply-bpe -c codes < ../train.article.txt | subword-nmt get-vocab > voc.article
#subword-nmt apply-bpe -c codes < ../train.title.txt | subword-nmt get-vocab > voc.title
#subword-nmt apply-bpe -c codes --vocabulary voc.article --vocabulary-threshold 50 < ../train.article.txt > ./train.article.BPE.txt
#subword-nmt apply-bpe -c codes --vocabulary voc.title --vocabulary-threshold 50 < ../train.title.txt > ./train.title.BPE.txt


#CollectVocab

# subword-nmt apply-bpe -c ../../train/subword/codes --vocabulary ../../train/subword/voc.article  --vocabulary-threshold 50 < ../valid.article.txt > ./valid.article.txt
# subword-nmt apply-bpe -c ../../train/subword/codes --vocabulary ../../train/subword/voc.title --vocabulary-threshold 50 < ../valid.title.txt > valid.title.txt
# sed -r 's/(@@ )|(@@ ?$)//g' out.txt > out_detoken.txt
