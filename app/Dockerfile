FROM python:3.8
WORKDIR /usr/local/bin


RUN apt-get update && apt-get install -y \
    sudo \
    pandoc \
    pandoc-citeproc \
    libcurl4-gnutls-dev \
    libcairo2-dev \
    libxt-dev \
    libssl-dev \
    libssh2-1-dev 

RUN apt-get update && \
    apt-get upgrade -y


COPY . .
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install pickle-mixin

EXPOSE 8050

CMD ["python","app.py"]

