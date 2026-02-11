FROM cupy/cupy:v13.6.0

RUN apt update -y && apt upgrade -y && apt autoremove -y

RUN pip install pytest

WORKDIR /mnt
ENV PYTHONPATH $PYTHONPATH:/mnt/src
CMD /bin/bash

