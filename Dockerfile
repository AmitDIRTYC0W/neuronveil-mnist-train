FROM python:3.11-bookworm

WORKDIR /usr/src

RUN pip install --no-cache-dir hatch

COPY pyproject.toml .
RUN hatch dep show requirements > requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install .

ENTRYPOINT ["neuronveil-mnist"]

