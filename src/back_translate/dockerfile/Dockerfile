FROM tensorflow/tensorflow:1.13.2-gpu
RUN pip install --user absl-py nltk==3.2 tensor2tensor==1.8.0
ENV PATH "$PATH:~/.local/bin"
RUN python -c "import nltk; nltk.download('punkt')"
WORKDIR /workspace
