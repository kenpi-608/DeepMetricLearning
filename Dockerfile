FROM ubuntu:18.04

RUN apt-get update && \
    apt-get install -y software-properties-common tzdata && \
    # リポジトリを追加
    apt-add-repository -y ppa:git-core/ppa && \
    # 新たにリポジトリを追加したのでまた更新
    apt-get update && \
    apt-get install -y git vim && \
    apt-get install -y fonts-ipaexfont
    # imageのサイズを小さくするためにキャッシュ削除
    # apt-get clean && \
    # rm -rf /var/lib/apt/lists/* && \
    


RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.bashrc

RUN apt-get install -y make build-essential libssl-dev zlib1g-dev \
                       libbz2-dev libreadline-dev libsqlite3-dev wget \
                       curl llvm libncurses5-dev xz-utils liblzma-dev \
                       tk-dev libxml2-dev libxmlsec1-dev libffi-dev

# PATHにpyenvを追加
ENV PATH $PATH:/root/.pyenv/bin

RUN pyenv install 3.8.5 && \
    # docker上でpythonのversionの切替はしないと思ったので、globalに
    pyenv global 3.8.5 && \
    pyenv rehash

ENV PATH $PATH:/root/.pyenv/shims

RUN pip install --upgrade pip

# 作業するディレクトリを変更
WORKDIR /home/DeepMetricLearning

# ホストにあるrequirements.txtをコンテナ内にコピー
COPY requirements.txt ${PWD}

# pythonのパッケージをインストール
RUN pip install -r requirements.txt

# Jupyter Notebook configの設定を行う
ENV CONFIG /root/.jupyter/jupyter_notebook_config.py
ENV CONFIG_IPYTHON /root/.ipython/profile_default/ipython_config.py

RUN jupyter notebook --generate-config --allow-root && \
    ipython profile create

RUN echo "c.NotebookApp.ip = '*'" >>${CONFIG} && \
    echo "c.NotebookApp.port = 9090" >>${CONFIG} && \
    echo "c.NotebookApp.open_browser = False" >>${CONFIG} && \
    echo "c.NotebookApp.allow_root = True" >>${CONFIG} && \
    echo "c.NotebookApp.iopub_data_rate_limit=10000000000" >>${CONFIG} && \
    echo "c.MultiKernelManager.default_kernel_name = 'python3'" >>${CONFIG} && \
    echo "c.InteractiveShellApp.exec_lines = ['%matplotlib inline']" >>${CONFIG_IPYTHON}

# port
EXPOSE 9090