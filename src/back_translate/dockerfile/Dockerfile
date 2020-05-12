FROM tensorflow/tensorflow:1.13.2-gpu
RUN apt-get install sudo
RUN sudo pip install --user absl-py nltk==3.2 tensor2tensor==1.2.1
RUN python -c "import nltk; nltk.download('punkt')"
WORKDIR /workspace
