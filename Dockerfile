
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
#FROM python:3.12
WORKDIR /app

COPY requirements.txt /tmp/requirements.txt

#RUN pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128

RUN pip install -r /tmp/requirements.txt
#COPY . .
CMD ["python", "main.py","-a", "NewBigModel","--lr","0.00005","--max-epoch", "2500", "--lossf", "fflMix","--train-batch-size","1","--use-dwt", "Wavelet"]


