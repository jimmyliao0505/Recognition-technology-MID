import cv2
import numpy as np

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    region = np.array([[(
        width // 4, height * 3 // 4),
        (width * 3 // 4, height * 3 // 4),
        (width * 3 // 4, height),
        (width // 4, height)
    ]], dtype=np.int32)
    cv2.fillPoly(mask, region, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=200)
    left_lines, right_lines = [], []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1 + 1e-6)

            if slope < -0.5:
                y1 = height
                x1 = width // 2 - 350
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0.5:
                y2 = height
                x2 = width // 2 + 230
                right_lines.append((x1, y1, x2, y2))

    lane_overlay = np.zeros_like(frame)
    for line in left_lines + right_lines:
        x1, y1, x2, y2 = line
        cv2.line(lane_overlay, (x1, y1), (x2, y2), (0, 255, 0), 3)

    if left_lines and right_lines:
        left_points = [(x1, y1) for x1, y1, x2, y2 in left_lines] + [(x2, y2) for x1, y1, x2, y2 in reversed(left_lines)]
        right_points = [(x1, y1) for x1, y1, x2, y2 in right_lines] + [(x2, y2) for x1, y1, x2, y2 in reversed(right_lines)]
        lane_area = np.array(left_points + right_points, dtype=np.int32)
        cv2.fillPoly(lane_overlay, [lane_area], (0, 255, 0))

    frame = cv2.addWeighted(lane_overlay, 0.5, frame, 1, 0)

    lane_center_x = width // 2
    if left_lines and right_lines:
        left_x = np.mean([line[0] for line in left_lines] + [line[2] for line in left_lines])
        right_x = np.mean([line[0] for line in right_lines] + [line[2] for line in right_lines])
        lane_center_x = int((left_x + right_x) / 2)

    vehicle_center_x = width // 2
    cv2.line(frame, (lane_center_x, height - 50), (lane_center_x, height - 30), (255, 0, 0), 3)
    cv2.line(frame, (vehicle_center_x, height - 50), (vehicle_center_x, height - 30), (0, 0, 255), 3)

    return frame


def main(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    # 取得原影片的資訊
    fps = int(cap.get(cv2.CAP_PROP_FPS))             # 影格率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 寬
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # 高

    # === 新增：建立影片輸出物件 ===
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 可用 'XVID' 或 'mp4v'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)

        # 顯示影片
        cv2.imshow("Lane Detection", processed_frame)

        # === 新增：寫入處理後的每一幀 ===
        out.write(processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # === 釋放資源 ===
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# 執行程式
video_path = "LaneVideo.mp4"
output_path = "LaneOutput.mp4"  # 你想要輸出的影片名稱
main(video_path, output_path)
