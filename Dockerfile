# Базовый образ: PyTorch + CUDA 12.1 Runtime
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# Немного утилит
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Клонируем TUTA
RUN git clone https://github.com/microsoft/TUTA_table_understanding.git tuta

WORKDIR /workspace/tuta

# Обновляем pip (можно и не трогать, но пусть будет ок)
RUN pip install --upgrade pip

# ВАЖНО: выкидываем строку python>=3.8.0 из requirements.txt,
# чтобы pip не пытался установить несуществующий пакет "python"
RUN sed -i '/^python[>= ]/d' requirements.txt

# Ставим зависимости TUTA
RUN pip install -r requirements.txt

# Рабочая директория по умолчанию
WORKDIR /workspace

CMD ["bash"]
