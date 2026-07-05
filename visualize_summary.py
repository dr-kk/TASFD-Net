import os
import cv2

VIDEO = "./datasets/Accident/v26.mp4"
SUMMARY = "./results/summaries/v26.mp4"

FIGURE_DIR = "./results/figures"
os.makedirs(FIGURE_DIR, exist_ok=True)


def save_frame(video_path, percent, output_name):

    print(f"\nOpening : {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("ERROR : Cannot open video")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("Total Frames :", total)

    frame_no = int(total * percent)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)

    ret, frame = cap.read()

    if not ret:
        print("ERROR : Cannot read frame", frame_no)
        cap.release()
        return

    output = os.path.join(FIGURE_DIR, output_name)

    ok = cv2.imwrite(output, frame)

    print("Saved :", ok, output)

    cap.release()


save_frame(VIDEO, 0.25, "original_25.jpg")
save_frame(VIDEO, 0.50, "original_50.jpg")
save_frame(VIDEO, 0.75, "original_75.jpg")

save_frame(SUMMARY, 0.25, "summary_25.jpg")
save_frame(SUMMARY, 0.50, "summary_50.jpg")
save_frame(SUMMARY, 0.75, "summary_75.jpg")

print("\nDone.")