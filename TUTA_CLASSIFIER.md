мой вопро: "давай вернемся к tuta и к тому как запустить их классификатор
https://github.com/microsoft/TUTA_table_understanding/tree/main/tuta"

ответ: "напиши мне скрипт который возьмет обработанные таблицы (уже в их формате .pt после запуска prepare.py)"

примет аргумент путь до модели
путь до папок с файлами .pt
и в папку output выдаст мне json
ChatGPT сказал:
Thought for 5m 11s
ChatGPT сказал:

Окей, давай сделаем понятный скелет скрипта для инференса CTC (cell type classification) поверх твоих .pt, который:

принимает:

--model_path – путь до чекпоинта TUTA (классификатор ячеек),

--data_dir – папка с .pt, которые ты получил через prepare.py,

--output_dir – куда складывать JSON’ы;

для каждого .pt:

прогоняет таблицы через модель,

пишет *.json с предсказаниями по ячейкам.

Сразу честно: из‑за ограничений я не могу сейчас прочитать исходники tuta/ и точно сказать, как называются все классы/ключи в .pt. Поэтому ниже – рабочий шаблон, который тебе нужно будет чуть‑чуть подогнать:

поправить import‑ы под реальные модули в tuta/,

проверить, как устроен dataset.pt (какие там ключи),

подставить правильный класс модели (там точно есть что‑то вроде TutaForCellClassification/TutaModel + голова).

Но общая структура скрипта будет именно такой.

Скрипт run_tuta_ctc_infer.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ==============================
# 1. Импорты из TUTA
# ==============================
#
# ⚠️ Тут тебе нужно посмотреть в репозитории `tuta/`,
# какие именно модули/классы используются для CTC.
#
# Ниже — пример, как это обычно выглядит.
#
# Пример: в tuta может быть что-то типа:
#   from model import TutaForCellClassification
#   from dataset import SpreadSheetCTCDataset
#
# Заменишь эти импорты на реальные.

# TODO: подставь свои пути/классы
# from model import TutaForCellClassification
# from dataset_ctc import SpreadSheetCTCDataset

# Если удобнее, можно вообще не использовать их Dataset,
# а просто руками читать torch.load(...) — пример ниже я дам.


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TUTA cell type classifier on prepared .pt tables and dump JSON"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Путь до чекпоинта TUTA (классификатор ячеек, .pt / .bin)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Папка с .pt файлами (output_path из prepare.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Папка, куда писать JSON с предсказаниями"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Размер батча по таблицам (обычно 1 удобно)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda или cpu"
    )
    return parser.parse_args()


# ==============================
# 2. Пример: как загрузить модель
# ==============================

def load_model(model_path: str, device: torch.device):
    """
    Загрузка TUTA + головы для CTC.

    Вариант 1: в чекпоинте сохранён целый объект модели:
        torch.save(model, path)
    Тогда:
        model = torch.load(path, map_location=device)

    Вариант 2: в чекпоинте только state_dict:
        torch.save({"model_state_dict": model.state_dict(), ...}, path)
    Тогда надо:
        state = torch.load(path)
        model = TutaForCellClassification(...)
        model.load_state_dict(state["model_state_dict"])
    """

    # --- Попытка 1: загрузить целиком объект модели ---
    state = torch.load(model_path, map_location=device)

    if isinstance(state, torch.nn.Module):
        model = state
    elif isinstance(state, dict) and "model_state_dict" in state:
        # TODO: здесь нужен правильный класс модели из TUTA
        # model = TutaForCellClassification(...)
        raise RuntimeError(
            "В чекпоинте только state_dict. "
            "Подставь здесь конструктор модели TUTA и раскомментируй."
        )
        # model.load_state_dict(state["model_state_dict"])
    else:
        raise RuntimeError(
            f"Не понимаю формат чекпоинта: {model_path}. "
            f"Ожидал либо целый nn.Module, либо dict с 'model_state_dict'."
        )

    model.to(device)
    model.eval()
    return model


# ==============================
# 3. Минимальный Dataset вокруг .pt
# ==============================

class SimpleTutaPtDataset(torch.utils.data.Dataset):
    """
    Очень простой враппер вокруг .pt, если не хочется
    разбираться с их Dataset-классами.

    Ожидаем, что .pt — это список / словарь с таблицами
    в формате, с которым уже умеет работать модель
    (как это делает их train-скрипт).
    """

    def __init__(self, pt_path: str):
        super().__init__()
        self.pt_path = pt_path
        data = torch.load(pt_path)

        # ⚠️ ТУТ НАДО ОДИН РАЗ ПРИНТАНУТЬ data у себя,
        # чтобы понять, что внутри, и поправить этот код.
        #
        # Примеры вариантов:
        #   1) data = {"tables": [...]}                → self.samples = data["tables"]
        #   2) data = list_of_tables                   → self.samples = data
        #   3) data = {"data": [...], "meta": ...}     → self.samples = data["data"]

        if isinstance(data, dict) and "tables" in data:
            self.samples = data["tables"]
        elif isinstance(data, list):
            self.samples = data
        else:
            # Временный вариант — просто сохраним всё как есть.
            # Ты потом подправишь под реальную структуру.
            self.samples = data

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Обычно элемент — это уже dict с тензорами/фичами:
        # {
        #   "input_ids": ...,
        #   "attention_mask": ...,
        #   "cell_index": ...,
        #   ...
        # }
        return self.samples[idx]


def default_collate(batch):
    """
    Наивный collate: предполагаем, что каждый элемент batch — dict с тензорами.
    torch.utils.data.default_collate с dict'ами умеет работать сам, так что
    можно просто его использовать.

    Если в TUTA есть свой collator (например, чтобы паддить таблицы),
    лучше импортнуть его и использовать вместо этого.
    """
    return torch.utils.data.default_collate(batch)


# ==============================
# 4. Запуск инференса по одному .pt
# ==============================

def run_inference_on_pt(model, dataset_path: str, output_path: str,
                        device: torch.device, batch_size: int = 1,
                        id2label=None):
    """
    Прогоняем один .pt (набор таблиц) через модель и сохраняем JSON.

    Формат JSON можно адаптировать под себя. Пример:
    [
      {
        "table_index": 0,
        "cells": [
          {"row": 0, "col": 0, "pred_label": "HEADER"},
          {"row": 0, "col": 1, "pred_label": "HEADER"},
          ...
        ]
      },
      ...
    ]
    """

    dataset = SimpleTutaPtDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, collate_fn=default_collate)

    all_tables = []

    with torch.no_grad():
        for table_idx, batch in enumerate(dataloader):
            # batch — либо dict с тензорами, либо что-то более хитрое.
            # Его нужно привести к тому виду, который ожидает
            # forward модели TUTA.

            # Пример: если модель принимает **batch:
            #   logits = model(**{k: v.to(device) for k, v in batch.items()})  # [B, N_cells, num_labels]

            if isinstance(batch, dict):
                batch_on_device = {
                    k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                    for k, v in batch.items()
                }
            else:
                # Если у тебя свой формат, тут его адаптируешь.
                raise RuntimeError(
                    "batch не dict. Подправь collate и этот кусок под реальный формат."
                )

            outputs = model(**batch_on_device)

            # ⚠️ Здесь нужно знать, что именно отдаёт model(...)
            #
            # Обычно для классификации это либо:
            #   outputs.logits  [B, N_cells, num_labels]
            # либо просто тензор.
            #
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs  # надеемся, что это уже тензор

            # Предположим размер [B, N_cells, num_labels]
            # Если B>1, тебе надо будет аккуратно разворачивать.
            # Для простоты считаем, что batch_size=1.
            logits = logits.squeeze(0)  # [N_cells, num_labels]

            probs = torch.softmax(logits, dim=-1)
            pred_ids = probs.argmax(dim=-1).tolist()

            # Теперь нужно сопоставить этим id координаты ячеек.
            # Обычно в batch есть что-то вроде:
            #   batch_on_device["cell_coord"] или ["row_ids"] / ["col_ids"]
            #
            # Я покажу общий шаблон, дальше подправишь под реальные ключи.

            if "cell_coord" in batch:
                coords = batch["cell_coord"]  # например, список (row, col)
                coords = [tuple(c) for c in coords]
            else:
                # Если координат нет, можно хотя бы вернуть индексы cell_idx
                coords = [(i, None) for i in range(len(pred_ids))]

            table_json = {
                "table_index": table_idx,
                "cells": []
            }
            for (row, col), pred_id in zip(coords, pred_ids):
                if id2label is not None:
                    label_name = id2label.get(int(pred_id), str(pred_id))
                else:
                    label_name = int(pred_id)

                table_json["cells"].append(
                    {
                        "row": int(row) if row is not None else None,
                        "col": int(col) if col is not None else None,
                        "pred_label": label_name
                    }
                )

            all_tables.append(table_json)

    # Пишем JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_tables, f, ensure_ascii=False, indent=2)


# ==============================
# 5. main: обходим все .pt и гоняем инференс
# ==============================

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = load_model(args.model_path, device)

    # Попробуем достать id2label из модели (часто так делают)
    id2label = None
    if hasattr(model, "config") and hasattr(model.config, "id2label"):
        id2label = model.config.id2label

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(p for p in data_dir.glob("*.pt"))
    if not pt_files:
        raise RuntimeError(f"В {data_dir} не найдено ни одного .pt файла")

    for pt_path in pt_files:
        out_name = pt_path.stem + ".json"
        out_path = output_dir / out_name
        print(f"[INFO] Обрабатываю {pt_path} → {out_path}")

        run_inference_on_pt(
            model=model,
            dataset_path=str(pt_path),
            output_path=str(out_path),
            device=device,
            batch_size=args.batch_size,
            id2label=id2label,
        )

    print("[DONE] Всё обработано.")


if __name__ == "__main__":
    main()

Что тебе нужно будет поправить под себя

Импорты модели / датасета

 

Вверху скрипта:

# from model import TutaForCellClassification
# from dataset_ctc import SpreadSheetCTCDataset


Зайди в /workspace/tuta и посмотри:

какие есть .py файлы,

как называется класс модели для CTC (что-то типа TutaForCellClassification),

есть ли готовый Dataset для CTC.

Если есть готовый Dataset и collate, намного лучше использовать их, а не SimpleTutaPtDataset.

Формат чекпоинта

 

В load_model() я заложил две ветки:

чекпоинт = целый nn.Module (тогда всё просто),

чекпоинт = dict с "model_state_dict" (тогда нужно создать модель и залить state_dict).

Посмотри, как у них сохраняются модели (в train‑скрипте) и подгони.

Формат .pt датасета

 

В SimpleTutaPtDataset я делаю:

data = torch.load(pt_path)
if isinstance(data, dict) and "tables" in data:
    self.samples = data["tables"]
elif isinstance(data, list):
    self.samples = data


Один раз у себя запусти в интерпретаторе:

import torch
d = torch.load("dataset.pt")
print(type(d))
print(d.keys() if isinstance(d, dict) else None)


И поправь __init__/__getitem__ под реальный формат.

Координаты ячеек

 

В run_inference_on_pt() я условно использую batch["cell_coord"].
У тебя в батче могут быть:

batch["row_ids"], batch["col_ids"],

или вообще отдельная структура.

Тебе нужно:

вытащить из батча список координат (row, col) в том же порядке, в каком модель выдаёт logits;

иначе можно хотя бы выводить индекс ячейки, а координаты маппить отдельно.