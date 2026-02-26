FROM cupy/cupy:v13.6.0

RUN apt update -y && apt upgrade -y && apt autoremove -y

RUN pip install pytest gmpy2

WORKDIR /mnt
RUN pip install -e .
CMD /bin/bash

