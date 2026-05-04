import custom_modules
from multiprocessing import freeze_support
from ultralytics import YOLO
import wandb

RESUME = True   # set to True to resume from last checkpoint
MODEL_PATH = "runs\\incskip96\\weights\\last.pt"  # path to last checkpoint (if RESUME=True)
TINYDS_PATH = "./yamls/tinyinc_v3.yaml"  # path to tiny dataset (if needed)

def main():
    # ── Stage 1: Train at 96x96 ────────────────────────────────────
    if RESUME:
        model = YOLO(MODEL_PATH)  # load last checkpoint
        model.train(resume=True, wandb_id='incskip96')  # resume training with same args
    else:
        model = YOLO(TINYDS_PATH)  # load model from tiny dataset config
    
        model.train(
            data="person.yaml",
            imgsz=96,
            epochs=1000,
            workers=3,
            batch=64,
            classes=[0],
            optimizer="Adam",
            lr0=1e-3,
            lrf=0.001,
            cos_lr=True,
            project="runs",
            name="incskip96",
            exist_ok=True,
            wandb_id='incskip96',
        )

   

    # ── Stage 2: Export to int8 TFLite ───────────────────────────────
    # model_final = YOLO("runs/nasBest/weights/best.pt")  # load best checkpoint
    # model_final.export(
    #     format="tflite",
    #     imgsz=96,
    #     int8=True,
    #     data="person.yaml",  # calibration data
    #     nms=True,
    #     simplify=True,
    #     dynamic=False
    # )
    # import onnx
    # onnx_model = onnx.load("runs/inc_v2/weights/best.onnx")
    # onnx.checker.check_model(onnx_model)
    # print("ONNX model is valid.")
    # onnx.save(onnx_model, "inc_v2.onnx")


if __name__ == "__main__":
    freeze_support()
    main()
