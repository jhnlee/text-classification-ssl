FROM tensorflow/tensorflow:1.13.0rc0-gpu
RUN pip2 install --user absl-py nltk==3.2 tensor2tensor==1.13.4
RUN python -c "import nltk; nltk.download('punkt')"
WORKDIR /workspace
