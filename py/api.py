from typing import Union
from fastapi import FastAPI

from scanner import close_device, wait_for_finger_and_capture, open_device_resilient, save_bmp_via_dll
from enhance import best_rotation_similarity

app = FastAPI()

ref = cv2.imread(REFERENCE, cv2.IMREAD_GRAYSCALE)

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/scan")
def scan_item(item_id: int, q: Union[str, None] = None):
    h = None
    try:
        h, mode = open_device_resilient()
        print(f"Opened in {mode} mode. Place finger on the sensor …")
        img = wait_for_finger_and_capture(h, DEFAULT_ADDR, TIMEOUT_SECONDS)
        print(f"Captured {len(img)} bytes. Saving BMP → {OUTPUT_BMP}")
        
        print("Done.")

        best_rotation_similarity(ref, img)
        
    finally:
        close_device(h)

    return {"item_id": item_id, "q": q}