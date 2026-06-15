import os
import torch
import torchvision
from ultralytics import YOLO


def get_strict_match_count(boxes_a, boxes_b, iou_threshold=0.5):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return 0
    ious = torchvision.ops.box_iou(boxes_a, boxes_b).clone()
    match_count = 0

    while True:
        max_iou = torch.max(ious)

        if max_iou < iou_threshold:
            break

        max_idx = torch.argmax(ious)
        row = max_idx // ious.shape[1]
        col = max_idx % ious.shape[1]

        match_count += 1

        ious[row, :] = -1.0
        ious[:, col] = -1.0

    return match_count
model = YOLO("yoloe-26x-seg.pt")
model.set_classes(["person"])
original_dir = "./clean/"
haze_dir = "./regression_result_jpg_files/metro_pure_haze_inputs/"
dehaze_dir = "./regression_result_jpg_files/metro_pure_predictions/"

valid_extensions = ('.png', '.jpg', '.jpeg')
image_files = [f for f in os.listdir(original_dir) if f.lower().endswith(valid_extensions)]

total_original_count = 0
haze_matched_count = 0
dehaze_matched_count = 0

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.6  # 박스가 50% 이상 겹쳐야 같은 사람으로 인정

print(f"총 {len(image_files)}개 이미지에 대해 '위치 기반 매칭(IoU)' 평가를 시작합니다...\n")

for img_file in image_files:
    orig_path = os.path.join(original_dir, img_file)
    haze_path = os.path.join(haze_dir, img_file.split(".")[0] + "_haze.png")
    dehaze_path = os.path.join(dehaze_dir, img_file.split(".")[0] + "_pred.png")

    res_orig = model.predict(source=orig_path, classes=[0], conf=CONF_THRESHOLD, verbose=False)
    res_haze = model.predict(source=haze_path, classes=[0], conf=CONF_THRESHOLD, verbose=False)
    res_dehaze = model.predict(source=dehaze_path, classes=[0], conf=CONF_THRESHOLD, verbose=False)

    boxes_orig = res_orig[0].boxes.xyxy
    boxes_haze = res_haze[0].boxes.xyxy
    boxes_dehaze = res_dehaze[0].boxes.xyxy

    total_original_count += len(boxes_orig)


    if len(boxes_orig) > 0 and len(boxes_haze) > 0:

        ious = torchvision.ops.box_iou(boxes_orig, boxes_haze)

        max_ious, _ = ious.max(dim=1)

        haze_matched_count += (max_ious > IOU_THRESHOLD).sum().item()

    if len(boxes_orig) > 0 and len(boxes_dehaze) > 0:
        ious = torchvision.ops.box_iou(boxes_orig, boxes_dehaze)
        max_ious, _ = ious.max(dim=1)
        dehaze_matched_count += (max_ious > IOU_THRESHOLD).sum().item()
for img_file in image_files:
    orig_path = os.path.join(original_dir, img_file)
    haze_path = os.path.join(haze_dir, img_file.split(".")[0] + "_haze.jpg")
    dehaze_path = os.path.join(dehaze_dir, img_file.split(".")[0] + "_pred.jpg")

    res_orig = model.predict(source=orig_path, classes=[0], conf=CONF_THRESHOLD, verbose=False)
    res_haze = model.predict(source=haze_path, classes=[0], conf=CONF_THRESHOLD, verbose=False)
    res_dehaze = model.predict(source=dehaze_path, classes=[0], conf=CONF_THRESHOLD, verbose=False)

    boxes_orig = res_orig[0].boxes.xyxy
    boxes_haze = res_haze[0].boxes.xyxy
    boxes_dehaze = res_dehaze[0].boxes.xyxy

    total_original_count += len(boxes_orig)

    haze_matched_count += get_strict_match_count(boxes_orig, boxes_haze, IOU_THRESHOLD)
    dehaze_matched_count += get_strict_match_count(boxes_orig, boxes_dehaze, IOU_THRESHOLD)
if total_original_count == 0:
    print("경고: Original 이미지에서 탐지된 사람이 0명입니다.")
else:
    haze_retention = (haze_matched_count / total_original_count) * 100
    dehaze_retention = (dehaze_matched_count / total_original_count) * 100

    print("-" * 40)
    print("[위치 기반 매칭(IoU) 결과 요약]")
    print(f"Original: {total_original_count}명")
    print(f"Haze: {haze_matched_count}명")
    print(f"Dehaze: {dehaze_matched_count}명")
    print("-" * 40)
    print(f"Haze 객체 유지율   : {haze_retention:.2f}%")
    print(f"Dehaze 객체 유지율 : {dehaze_retention:.2f}%")