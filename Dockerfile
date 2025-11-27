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

# Ставим зависимости TUTA
WORKDIR /workspace/tuta

# Обновляем pip и ставим requirements
# Если в requirements.txt есть строка с torch, можно её закомментировать,
# чтобы не конфликтовала с уже установленным PyTorch из базового образа.
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Рабочая директория по умолчанию — корневая
WORKDIR /workspace

# По умолчанию просто открываем bash, чтобы ты мог руками запускать скрипты
CMD ["bash"]
