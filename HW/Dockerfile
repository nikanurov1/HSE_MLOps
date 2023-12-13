# Установка базового образа
FROM python:3.9

RUN wget https://dl.min.io/server/minio/release/linux-amd64/minio && \
    chmod +x minio && \
    mv minio /usr/local/bin/ && \
    wget https://dl.min.io/client/mc/release/linux-amd64/mc && \
    chmod +x mc && \
    mv mc /usr/local/bin/

# Установка рабочей директории внутри контейнера
WORKDIR /app

# Копирование requirements.txt и установка зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование init.sh и api.py внутрь контейнера
COPY src/ .
COPY init.sh .
COPY api.py .

# RUN chmod +x init.sh

# Запуск init.sh 
CMD ["sh", "init.sh"] && python api.py