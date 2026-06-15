from ultralytics import YOLO

# Load a model
model = YOLO("yoloe-26x-seg.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom model
model.set_classes(["person"])
# Predict with the model
# results = model("./regression_result_jpg_files/metro_pure_predictions/zhaji02_96_pred.jpg")  # predict on an image
results = model("./clean/zhaji02_96.jpg")

results[0].show()

# =================================

# from ultralytics import YOLOE
#
# model = YOLOE("yoloe-26x-seg.pt")
#
#
# model.set_classes(["person"])
#
#
# # source_folder = f"./regression_result_jpg_files/metro_pure_haze_inputs/"
# source_folder = f"./clean/"
# results = model.predict(
#     source=source_folder,
#     save=True,
#     project="./predictions",
#     name="clean_image_detect"
# )
#
# print("모든 이미지의 예측 및 저장이 완료되었습니다!")
# print("결과물 저장 위치: runs/predict/person_results")

# =========================================
# import os
# import cv2
# from ultralytics import YOLOE  # (또는 YOLO)
#
# # 1. 모델 로드 및 설정
# model = YOLOE("yoloe-26x-seg.pt")
# model.set_classes(["person"])
#
# # 2. 경로 설정
# source_folder = "./metro_pure_haze_inputs/metro_pure_predictions"
# output_folder = "./dehaze_image_detect_bottom_left"  # 저장 폴더 변경
#
# os.makedirs(output_folder, exist_ok=True)
# image_files = [f for f in os.listdir(source_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
# print(f"총 {len(image_files)}장의 이미지를 처리합니다...")
#
# # 3. 반복문을 돌면서 한 장씩 예측하고 텍스트를 추가하여 저장
# for img_file in image_files:
#     img_path = os.path.join(source_folder, img_file)
#
#     # YOLO 예측 실행
#     results = model.predict(source=img_path, save=False, verbose=False)
#     result = results[0]
#
#     person_count = len(result.boxes)
#     annotated_img = result.plot()
#
#     text = f"Persons: {person_count}"
#
#     # --- 위치 설정: 좌측 하단 ---
#     img_height, img_width = annotated_img.shape[:2]
#     # x 좌표는 20, y 좌표는 이미지 높이에서 30 픽셀 위로 설정 (여백 확보)
#     position = (20, img_height - 30)
#
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = 1.0
#     font_color = (0, 0, 255)
#     thickness = 2
#
#     cv2.putText(annotated_img, text, position, font, font_scale, font_color, thickness)
#
#     save_path = os.path.join(output_folder, img_file)
#     cv2.imwrite(save_path, annotated_img)
#
# print("\n✅ 좌측 하단 텍스트 추가 완료!")



