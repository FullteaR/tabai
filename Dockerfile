FROM cupy/cupy:v14.0.1

RUN apt update -y && apt upgrade -y && apt install -y python3-venv && apt autoremove -y

RUN python3 -m venv --system-site-packages /opt/venv
ENV PATH /opt/venv/bin:$PATH
RUN pip install pytest gmpy2

WORKDIR /mnt
ENV PYTHONPATH $PYTHONPATH:/mnt/src
CMD /bin/bash

