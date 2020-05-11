FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-devel
RUN git clone https://github.com/NVIDIA/apex.git && cd apex && python setup.py install --cuda_ext --cpp_ext
RUN pip install transformers sklearn nltk tensorboardX
RUN python -c "import nltk; nltk.download('stopwords')"
WORKDIR /workspace
