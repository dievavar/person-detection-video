#!/usr/bin/env python3
"""Entry point: читать видео → детектить людей → рисовать → сохранять."""

import argparse
from src.utils import open_video, read_video_props, create_writer
from src.infer import load_model, detect_persons
from src.drawing import draw_box

def parse_args() -> argparse.Namespace:
    """Принимает аргументы командной строки для скрипта детекции людей.

    Функция читает флаги, с которыми запущен скрипт, и возвращает объект
    с готовыми полями.
    Эти значения далее используются при чтении/записи видео и запуске модели.

    Доступные флаги:
      --input Путь к входному видео (по умолчанию: crowd.mp4).
      --output Путь к выходному видео с разметкой (out/crowd_annotated.mp4).
      --model Весы модели YOLO, например yolov8n.pt или yolov8s.pt.
      --device Устройство: auto / cuda / cpu (по умолчанию: auto).
      --conf Порог уверенности детекций [0..1], ниже — отбрасываем.
      --iou Порог IoU для NMS [0..1], управляет слиянием перекрытий.
      --stride Обрабатывать каждый N-й кадр.
      --min_box Отбрасывать слишком маленькие боксы (минимальная сторона, px).
      --imgsz Входной размер для инференса (px).
      --max_det Лимит детекций на кадр.

    Возвращает:
        argparse.Namespace: Объект с разобранными параметрами, который
        передаётся дальше в код (args.*).
    """
    p = argparse.ArgumentParser(
        description="Детекция людей на видео."
    )
    p.add_argument(
        "--input",
        type=str,
        default="crowd.mp4",
        help="Путь к входному видео.",
    )
    p.add_argument(
        "--output",
        type=str,
        default="out/crowd_annotated.mp4",
        help="Путь к выходному MP4.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8s.pt",
        help="Весы YOLO.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Устройство инференса: auto/cuda/cpu.",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Порог уверенности детекций.",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.55,
        help="Порог IoU для выявления самого уверенного бокса.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Обрабатывать каждый N-й кадр для ускорения.",
    )
    p.add_argument(
        "--min_box",
        type=int,
        default=1,
        help="Отбрасывать боксы меньше указанной стороны (px).",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Входной размер для инференса YOLO (px)."
    )
    p.add_argument(
        "--max_det",
        type=int,
        default=1000,
        help="Лимит детекций на кадр.")

    return p.parse_args()


def main() -> None:
    """Главная точка входа: прочитать видео → детектировать людей → отрисовать → сохранить.

    Последовательность действий:
      1) Разобрать аргументы командной строки (`parse_args`).
      2) Открыть входное видео и считать его свойства (FPS, ширину, высоту).
      3) Создать видеописатель под выходной MP4 с тем же FPS и размером.
      4) Загрузить модель YOLO и определить устройство инференса (CPU/CUDA/auto).
      5) Для каждого кадра:
         - выполнить детекцию людей;
         - отрисовать рамки и подписи `person {score:.2f}`;
         - записать кадр в выходное видео.
      6) Освободить ресурсы и вывести путь к результату.

    Args:
        None

    Returns:
        None

    Side Effects:
        - Создаёт/перезаписывает файл выходного видео по пути `--output`.
        - Печатает в stdout строку с путём к сохранённому файлу.

    Raises:
        FileNotFoundError: если входное видео нельзя открыть в `open_video`.
        RuntimeError: если не удаётся создать `VideoWriter` в `create_writer`.
        ValueError: если некорректные CLI-параметры (на этапе `parse_args` или валидации).
    """
    args = parse_args()

    cap = open_video(args.input)
    fps, w, h = read_video_props(cap)

    writer = create_writer(args.output, fps, (w, h))

    model, device = load_model(args.model, args.device)

    # Основной цикл по кадрам
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if idx % args.stride == 0:
            dets = detect_persons(
                model=model,
                frame=frame,
                conf=args.conf,
                iou=args.iou,
                min_box=args.min_box,
                imgsz=args.imgsz,
                max_det=args.max_det
            )
            for x1, y1, x2, y2, conf in dets:
                draw_box(frame, (x1, y1, x2, y2), f"person {conf:.2f}", score=conf)

        writer.write(frame)
        idx += 1

    cap.release()
    writer.release()
    print(f"Сделано: {args.output}")


if __name__ == "__main__":
    main()
