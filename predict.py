import custom_modules
from ultralytics import YOLO
import cv2

IMG_PATH = "D:\\VidSumN\\Old\\images.jpg"

def main():
    model = YOLO("D:\\VideoSummarizer\\TinyissimoYOLO\\runs\\inceptionYOLO\\weights\\best.pt")
    results = model(IMG_PATH)
    for r in results:
        print(r.boxes.xyxy)
        print(r.boxes.conf)
        print(r.boxes.cls)
        img = r.plot()
        cv2.imshow("img", img)
        cv2.waitKey(0)
    

if __name__ == "__main__":
    main()
