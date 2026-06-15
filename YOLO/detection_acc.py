import os
from ultralytics import YOLO

model = YOLO("yoloe-26x-seg.pt")
model.set_classes(["person"])

original_dir = "./clean/"
haze_dir = "./regression_result_jpg_files/metro_pure_haze_inputs/"
dehaze_dir = "./regression_result_jpg_files/metro_pure_predictions/"

valid_extensions = ('.png', '.jpg', '.jpeg')
image_files = [f for f in os.listdir(original_dir) if f.lower().endswith(valid_extensions)]

total_original_count = 0
total_haze_count = 0
total_dehaze_count = 0

CONF_THRESHOLD = 0.5

print(f"총 {len(image_files)}개 이미지에 대해 '단순 탐지 수(Count)' 평가를 시작합니다...\n")

for img_file in image_files:
    base_name = img_file.split(".")[0]

    orig_path = os.path.join(original_dir, img_file)
    haze_path = os.path.join(haze_dir, base_name + "_haze.jpg")
    dehaze_path = os.path.join(dehaze_dir, base_name + "_pred.jpg")

    if not (os.path.exists(haze_path) and os.path.exists(dehaze_path)):
        print(f"경고: {base_name} 에 해당하는 haze 또는 dehaze 파일이 없습니다. 건너뜁니다.")
        continue


    res_orig = model.predict(source=orig_path, classes=[0], conf=CONF_THRESHOLD, verbose=False)
    res_haze = model.predict(source=haze_path, classes=[0], conf=CONF_THRESHOLD, verbose=False)
    res_dehaze = model.predict(source=dehaze_path, classes=[0], conf=CONF_THRESHOLD, verbose=False)

    total_original_count += len(res_orig[0].boxes)
    total_haze_count += len(res_haze[0].boxes)
    total_dehaze_count += len(res_dehaze[0].boxes)

if total_original_count == 0:
    print("경고: Original 이미지에서 탐지된 사람이 0명입니다.")
else:
    # 전체 탐지 수를 기준으로 비율 계산
    haze_detection_rate = (total_haze_count / total_original_count) * 100
    dehaze_detection_rate = (total_dehaze_count / total_original_count) * 100

    print("-" * 40)
    print("[단순 사람 탐지 수(Count) 결과 요약]")
    print(f"Original 총 탐지 수 : {total_original_count}명")
    print(f"Haze 총 탐지 수     : {total_haze_count}명")
    print(f"Dehaze 총 탐지 수   : {total_dehaze_count}명")
    print("-" * 40)
    print(f"Original 대비 Haze 탐지율   : {haze_detection_rate:.2f}%")
    print(f"Original 대비 Dehaze 탐지율 : {dehaze_detection_rate:.2f}%")