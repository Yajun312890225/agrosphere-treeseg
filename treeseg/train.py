"""训练 YOLO 实例分割模型（Ultralytics）。

示例：
  python -m treeseg.train --data dataset/data.yaml --model yolo11m-seg.pt --epochs 100 --imgsz 1024 --batch 8
"""
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="yolo11m-seg.pt", help="预训练权重：yolo11n/s/m/l/x-seg.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8, help="-1 为自动按显存选择")
    ap.add_argument("--device", default=None, help="0 / 0,1 / cpu / mps，缺省自动")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="treeseg")
    ap.add_argument("--fraction", type=float, default=1.0, help="只用部分训练集（冒烟测试）")
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--mask-ratio", type=int, default=4,
                    help="掩膜下采样倍率，掩膜原型分辨率固定为 imgsz/4，设为 2 只会多耗 4 倍显存（A10 24G 上会 OOM）")
    ap.add_argument("--nbs", type=int, default=0,
                    help="名义批量（梯度累积目标）。ultralytics 默认 64，小数据集下每轮只更新 batch/64 次，几乎学不动；0 表示等于 batch，不做累积")
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model)
    kw = dict(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, workers=a.workers, project=a.project, name=a.name,
              fraction=a.fraction, patience=a.patience, exist_ok=True,
              # 航拍树冠没有方向性：加上下翻转与旋转增强；关闭 mixup 避免冠形失真
              flipud=0.5, fliplr=0.5, degrees=90.0, mosaic=1.0, mixup=0.0, hsv_h=0.01, hsv_s=0.4, hsv_v=0.3,
              overlap_mask=False, mask_ratio=a.mask_ratio,
              nbs=a.nbs if a.nbs > 0 else max(a.batch, 1))
    if a.device is not None:
        kw["device"] = a.device
    if a.resume:
        kw["resume"] = True
    r = model.train(**kw)
    print("最优权重:", getattr(r, "save_dir", a.project), "/weights/best.pt")


if __name__ == "__main__":
    main()
