# One Reference Is Not Enough: Diverse Distillation with Reference Selection for Non-Autoregressive Translation
This repository contains the source code for our NAACL 2022 main conference paper One Reference Is Not Enough: Diverse Distillation with Reference Selection for Non-Autoregressive Translation [pdf](https://arxiv.org/pdf/2205.14333.pdf). This code is implemented based on the open-source toolkit [fairseq-0.10.2](https://github.com/pytorch/fairseq).

# Requirements
This system has been tested in the following environment.

+ Python version = 3.8
+ Pytorch version = 1.7

# Diverse Distillation
Perform diverse distillation to obtain a dataset containing multiple references. You can follow the instructions below to prepare the diverse distillation dataset for WMT14 En-De. Or you can directly download [our diverse distillation dataset](todo) and jump to step 4.

**Step 1**: Follow instruction from [Fairseq](https://github.com/pytorch/fairseq/tree/master/examples/translation) to prepare and preprocess the WMT14 En-De dataset, or download the preprocessed dataset [here](http://dl.fbaipublicfiles.com/nat/original_dataset.zip). Save the raw data to ``data/wmt_ende`` (train.en-de.{en,de}, valid.en-de.{en,de}, test.en-de.{en,de}). Save the processed data to ``data-bin/wmt14_ende_raw``.

**Step 2**: Train 3 different autoregressive models by using 3 different seeds. 

```bash
data_dir=data-bin/wmt14_ende_raw
save_dir=output/wmt14_ende_at
for seed in {1..3}
do
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py $data_dir \
    --dropout 0.1 --fp16 --seed $seed --save-dir $save_dir$seed \
    --arch transformer_wmt_en_de  --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 --min-lr 1e-09 \
    --weight-decay 0.0 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --max-tokens 4096 --update-freq 1\
    --no-progress-bar --log-format json --log-interval 1000 --save-interval-updates 5000 \
    --max-update 150000 --keep-interval-updates 5 --keep-last-epochs 5
sh tools/average.sh $save_dir$seed
done
```
**Step 3**: use each model to decode the training set, obtain three decoding results pred.1, pred.2, pred.3.
```bash
data_dir=data-bin/wmt14_ende_raw
save_dir=output/wmt14_ende_at
for seed in {1..3}
do
CUDA_VISIBLE_DEVICES=0 python generate.py $data_dir --path $save_dir$seed/average-model.pt --gen-subset train --beam 5 --batch-size 100 --lenpen 0.6 > out.$seed
grep ^H out.$seed | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.$seed
done
```
**Step 4**: Concat the three decoding results with a special token \<divide>, and then preprocess the diverse distillation dataset.
```bash
data_dir=data/wmt14_ende
dest_dir=data-bin/wmt14_ende_divdis

python tools/concat.py
mv train.divdis.de $data_dir/
cp $data_dir/train.en-de.en $data_dir/train.divdis.en
python preprocess.py --source-lang en --target-lang de \
        --trainpref $data_dir/train.divdis \
        --validpref $data_dir/valid.en-de \
        --testpref $data_dir/test.en-de \
        --destdir $dest_dir \
        --joined-dictionary --workers 32\
```

# Reference Selection
Train a CTC model on the diverse distillation dataset with reference selection. We implement the loss functions in [nat_loss.py](https://github.com/ictnlp/DDRS-NAT/blob/main/fairseq/criterions/nat_loss.py).

**Step 1**: Apply reference selection to train the CTC model. Adjust --updata-freq if the number of GPU devices is not 8.
```bash
data_dir=data-bin/wmt14_ende_divdis
save_dir=output/wmt14ende_disdiv
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py $data_dir \
    --num-references 3 --ctc-ratio 3 --src-embedding-copy --fp16 --ddp-backend=no_c10d --save-dir $save_dir \
    --task translation_lev \
    --criterion ddrs_loss \
    --arch nonautoregressive_transformer \
    --noise full_mask \
    --optimizer adam --adam-betas '(0.9,0.98)'  \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --activation-fn gelu \
    --dropout 0.2 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'simple' --log-interval 1000 \
    --max-tokens 4096 --update-freq 1\
    --save-interval-updates 5000 \
    --max-update 300000 --keep-interval-updates 5 --keep-last-epochs 5
sh tools/average.sh $save_dir
```
**Step 2**: finetune the CTC model with max-reward reinforcement learning.
```bash
data_dir=data-bin/wmt14_ende_divdis
save_dir=output/wmt14ende_disdiv
cp $save_dir/average-model.pt ${save_dir}tune/checkpoint_last.pt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python train.py $data_dir \
    --tune --reset-optimizer --num-references 3 --ctc-ratio 3 --src-embedding-copy --fp16 --ddp-backend=no_c10d --save-dir ${save_dir}tune \
    --task translation_lev \
    --criterion ddrs_loss \
    --arch nonautoregressive_transformer \
    --noise full_mask \
    --optimizer adam --adam-betas '(0.9,0.98)'  \
    --lr 0.00002 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 500 \
    --warmup-init-lr '1e-07' --activation-fn gelu \
    --dropout 0.1 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.1 \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --max-tokens 4096 --update-freq 1\
    --save-interval-updates 500 \
    --max-update 3000 --keep-interval-updates 5 --keep-last-epochs 5
```
# Inference
**Step 1**: Decode the test set with argmax decoding.
```bash
model=output/wmt14ende_disdivtune/checkpoint_last.pt
data_dir=data-bin/wmt14_ende_divdis
CUDA_VISIBLE_DEVICES=0 python generate.py $data_dir \
    --gen-subset test \
    --task translation_lev \
    --iter-decode-max-iter  0  \
    --iter-decode-eos-penalty 0 \
    --path $model \
    --beam 1  \
    --left-pad-source False \
    --batch-size 100 > out
grep ^H out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.raw
python tools/dedup.py
python tools/deblank.py
sed -r 's/(@@ )|(@@ ?$)//g' pred.de.deblank > pred.de
perl tools/multi-bleu.perl ref.de < pred.de
```
**Step 2**: We can also apply beam search decoding combined with a 4-gram language model to search the target sentence. First, install the ctcdecode package.
```bash
git clone --recursive https://github.com/MultiPath/ctcdecode.git
cd ctcdecode && pip install .
```
Notice that it is important to install [MultiPath/ctcdecode](https://github.com/MultiPath/ctcdecode) rather than the original package. This version pre-computes the top-K candidates before running the beam-search, which makes the decoding much faster. Then, follow [kenlm](https://github.com/kpu/kenlm) to train a target-side 4-gram language model and save it as ``wmt14ende.arpa``. Finally, decode the test set with beam search decoding combined with a 4-gram language model.
```bash
model=output/wmt14ende_disdivtune/checkpoint_last.pt
data_dir=data-bin/wmt14_ende_divdis
CUDA_VISIBLE_DEVICES=0 python generate.py $data_dir \
    --use-beamlm \
    --beamlm-path ./wmt14ende.arpa \
    --alpha $1 \
    --beta $2 \
    --gen-subset test \
    --task translation_lev \
    --iter-decode-max-iter  0  \
    --iter-decode-eos-penalty 0 \
    --path $model \
    --beam 1  \
    --left-pad-source False \
    --batch-size 100 > out
grep ^H out | cut -f1,3- | cut -c3- | sort -k1n | cut -f2- > pred.raw
sed -r 's/(@@ )|(@@ ?$)//g' pred.raw > pred.de
perl tools/multi-bleu.perl ref.de < pred.de
```
The optimal choices of alpha and beta vary among datasets and can be found by grid-search.
# Citation

If you find the resources in this repository useful, please cite as:

``` bibtex
@inproceedings{ddrs,
  title = {One Reference Is Not Enough: Diverse Distillation with Reference Selection for Non-Autoregressive Translation},
  author= {Chenze Shao and Xuanfu Wu and Yang Feng},
  booktitle = {Proceedings of NAACL 2022},
  year = {2022},
}
```
