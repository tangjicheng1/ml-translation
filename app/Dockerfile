FROM huggingface/transformers-pytorch-gpu:4.21.0

LABEL maintainer="tangjch@hotmail.com"

# install tools
RUN apt-get update && apt-get install -y curl build-essential vim

SHELL ["/bin/bash", "-c"]

# copy application files
COPY ./*.py /app/
COPY ./requirements.txt /app/

# install python library 
RUN curl https://bootstrap.pypa.io/get-pip.py | python3
RUN pip install -r /app/requirements.txt

# convert M2M100 from pytorch model to onnx model
RUN optimum-cli export onnx -m facebook/m2m100_418M --for-ort /app/model

WORKDIR /app
EXPOSE 9527
CMD ["uvicorn", "api:app", "--host", "0.0.0.0" , "--port", "9527"]