import os
from ultralytics import YOLO
import custom_modules
os.environ["WANDB_MODE"] = "disabled"


def main():
    model = YOLO("temp.yaml")
    results = model.train(
                data     = "person.yaml",
                imgsz    = 128,
                epochs   = 2,
                batch    = 64,
                workers  = 3,
                optimizer= "Adam",
                lr0      = 1e-3,
                cos_lr   = True,
                project  = "best_arch",
                name     = f"nas",
                exist_ok = True,
                verbose  = False,
                save     = True,
                device   = "0",
            )

if __name__ == "__main__":
    main()