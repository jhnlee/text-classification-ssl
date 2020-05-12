# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/bin/bash

# Download transformer-big
folder=checkpoints
if [ ! -d ${folder} ]; then
  echo "*** Download Transformer Models ***"
  filename=back_trans_checkpoints.zip
  wget https://storage.googleapis.com/uda_model/text/${filename}
  unzip ${filename} && rm ${filename}
  mv ${folder}/vocab.translate_enfr_wmt32k.32768.subwords ${folder}/vocab.enfr.large.32768
fi


# replicas: An argument for parallel preprocessing. For example, when replicas=3,
# we divide the data into three parts, and only process one part
# according to the worker_id.

replicas=$3
worker_id=$2

# input_file: The file to be back translated. We assume that each paragraph is in
# a separate line

data=$1
input_file=input/${data}.txt

# sampling_temp: The sampling temperature for translation. See README.md for more
# details.

sampling_temp=0.9


# Dirs
data_dir=back_trans_data
doc_len_dir=output/${data}/doc_len
forward_src_dir=output/${data}/forward_src
forward_gen_dir=output/${data}/forward_gen
backward_gen_dir=output/${data}/backward_gen
para_dir=output/${data}/paraphrase

mkdir -p ${data_dir}
mkdir -p ${forward_src_dir}
mkdir -p ${forward_gen_dir}
mkdir -p ${backward_gen_dir}
mkdir -p ${doc_len_dir}
mkdir -p ${para_dir}

echo "*** spliting paragraph ***"
python split_paragraphs.py \
  --input_file=${input_file} \
  --output_file=${forward_src_dir}/file_${worker_id}_of_${replicas}.txt \
  --doc_len_file=${doc_len_dir}/doc_len_${worker_id}_of_${replicas}.json \
  --replicas=${replicas} \
  --worker_id=${worker_id} \

echo "*** forward translation ***"
t2t-decoder \
  --problem=translate_enfr_wmt32k \
  --model=transformer \
  --hparams_set=transformer_big \
  --hparams="sampling_method=random,sampling_temp=${sampling_temp}" \
  --decode_hparams="beam_size=1,batch_size=128" \
  --checkpoint_path=checkpoints/enfr/model.ckpt-500000 \
  --output_dir=/tmp/t2t \
  --decode_from_file=${forward_src_dir}/file_${worker_id}_of_${replicas}.txt \
  --decode_to_file=${forward_gen_dir}/file_${worker_id}_of_${replicas}.txt \
  --data_dir=checkpoints

echo "*** backward translation ***"
t2t-decoder \
  --problem=translate_enfr_wmt32k_rev \
  --model=transformer \
  --hparams_set=transformer_big \
  --hparams="sampling_method=random,sampling_temp=${sampling_temp}" \
  --decode_hparams="beam_size=1,batch_size=128,alpha=0" \
  --checkpoint_path=checkpoints/fren/model.ckpt-500000 \
  --output_dir=/tmp/t2t \
  --decode_from_file=${forward_gen_dir}/file_${worker_id}_of_${replicas}.txt \
  --decode_to_file=${backward_gen_dir}/file_${worker_id}_of_${replicas}.txt \
  --data_dir=checkpoints

echo "*** transform sentences back into paragraphs***"
python sent_to_paragraph.py \
  --input_file=${backward_gen_dir}/file_${worker_id}_of_${replicas}.txt \
  --doc_len_file=${doc_len_dir}/doc_len_${worker_id}_of_${replicas}.json \
  --output_file=${para_dir}/file_${worker_id}_of_${replicas}.json

