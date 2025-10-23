from typing import List, Tuple
import numpy as np
from ultralytics import YOLO

def load_model(weights: str, device: str | None):
    """Загрузить модель YOLO и определить устройство инференса.

      Функция инициализирует модель по указанным весам и возвращает саму модель,
      а также строку устройства, на котором её рекомендуется запускать.

      Args:
          weights (str): Путь или имя весов модели YOLO.
              - Имя из хаба Ultralytics ("yolov8n.pt"/"yolov8s.pt").
              - Локальный путь к файлу .pt.
          device (Optional[str]): Предпочитаемое устройство инференса.
              Допустимые значения:
              - ``"auto"`` или ``None`` — предоставить выбор Ultralytics.
              - ``"cuda"`` — принудительно использовать GPU (если доступна).
              - ``"cpu"`` — принудительно использовать процессор.

      Returns:
          Tuple[YOLO, Optional[str]]: Кортеж ``(model, chosen_device)``, где
              - ``model``: загруженная модель ``YOLO``.
              - ``chosen_device``: строка устройства для передачи в model.predict/
      """
    return YOLO(weights), (None if device == "auto" else device)


def detect_persons(
    model: YOLO,
    frame: np.ndarray,
    conf: float,
    iou: float,
    min_box: int,
    imgsz: int,
    max_det=int
) -> List[Tuple[int, int, int, int, float]]:
    """Выполнить детекцию людей на одном кадре и вернуть боксы с уверенностью.

    Запускает инференс YOLO на переданном кадре `frame`, применяя указанные
    пороги `conf` и `iou`, а также входной размер `imgsz`. После получения
    результата функция:
    1) фильтрует детекции по классу, оставляя только `'person'`;
    2) отбрасывает маленькие боксы, у которых min(ширина, высота) < `min_box`;
    3) возвращает список кортежей `(x1, y1, x2, y2, conf)` в пикселях кадра.

    Args:
        model (YOLO): Загруженная модель Ultralytics YOLO.
        frame (np.ndarray): Изображение кадра в формате BGR,
            размером H×W×3 и типом `uint8`.
        conf (float): Порог уверенности детекций [0..1]. Боксы со скором ниже
            порога будут отброшены внутри `model.predict`.
        iou (float): Порог IoU для NMS [0..1]. Используется при подавлении
            перекрывающихся боксов; управляет агрессивностью слияния.
        min_box (int): Минимальный размер стороны бокса (px). Если min(ширина,
            высота) меньше этого значения — бокс считается шумом и отбрасывается.
        imgsz (int): Входной размер для инференса (px). Кадр подгоняется к
            квадрату `imgsz×imgsz` (с паддингом), что улучшает поиск мелких
            объектов, но замедляет инференс и требует больше памяти.

    Returns:
        List[Tuple[int, int, int, int, float]]: Список детекций только класса
        `person`, каждая как `(x1, y1, x2, y2, conf)`, где координаты — целые
        пиксели в системе координат исходного кадра, `conf` — уверенность [0..1].
    """
    results = model.predict(
        source=frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        max_det= max_det,
        verbose=False,
    )
    r = results[0]
    out: List[Tuple[int, int, int, int, float]] = []

    # Если боксов нет — сразу возвращаем пустой список.
    if r.boxes is None or len(r.boxes) == 0:
        return out

    # Словарь имён классов (id -> name), чтобы отфильтровать только 'person'.
    names = r.names

    # Проходим по боксам: фильтруем класс, конвертируем координаты и применяем min_box.
    for b in r.boxes:
        cls_id = int(b.cls.item())
        if names.get(cls_id, "") != "person":
            continue

        # Координаты в формате xyxy (левая-верхняя и правая-нижняя точки).
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())

        # Отсеиваем слишком маленькие рамки (обычно это шум на фоне).
        if min(x2 - x1, y2 - y1) < min_box:
            continue

        conf_score = float(b.conf.item())
        out.append((x1, y1, x2, y2, conf_score))

    return out

