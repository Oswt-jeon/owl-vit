import torch
import cv2
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import os
import math

# -------------------------------
# 하이퍼파라미터
# -------------------------------
SCORE_THRESH   = 0.75   # 1차 컷
NMS_THRESH     = 0.3
TOPK           = None   # ← 우선 None으로 두고 실제 개수부터 보자. (필요하면 10~20으로)
MIN_AREA_RATIO = 0.003  # 0.3%로 상향 (장면에 따라 0.005까지도)
MAX_AR         = 4.0    # 종횡비 제한 강화
RADIUS_NMS_PX  = 48     # 중심거리 기반 추가 억제(해상도 따라 32~96로 조정)
DYN_ALPHA      = 1.0    # 동적 스코어 컷: mean + alpha*std

# -------------------------------
# 모델 설정 (Ensemble 해제)
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/owlv2-base-patch16"   # ← ensemble이 아닌 단일 모델

processor = Owlv2Processor.from_pretrained(model_id)
model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()

# -------------------------------
# 경로
# -------------------------------
target_img_path = "images/target3.jpeg"
query_img_path  = "images/query5.jpeg"
output_path     = "images/result.jpg"

if not os.path.exists(target_img_path):
    raise FileNotFoundError(f"❌ target 이미지가 존재하지 않습니다: {target_img_path}")
if not os.path.exists(query_img_path):
    raise FileNotFoundError(f"❌ query 이미지가 존재하지 않습니다: {query_img_path}")

# -------------------------------
# 로드
# -------------------------------
target_img = Image.open(target_img_path).convert("RGB")
query_img  = Image.open(query_img_path).convert("RGB")
H, W = target_img.height, target_img.width
IMG_AREA = float(H * W)
MIN_AREA = IMG_AREA * MIN_AREA_RATIO

# -------------------------------
# 추론
# -------------------------------
inputs = processor(
    images=target_img,
    query_images=[query_img],
    return_tensors="pt"
).to(device)

with torch.no_grad():
    outputs = model.image_guided_detection(**inputs)

res = processor.post_process_image_guided_detection(
    outputs=outputs,
    target_sizes=torch.tensor([(H, W)], device=device),
    threshold=SCORE_THRESH,
    nms_threshold=NMS_THRESH
)[0]

boxes: torch.Tensor  = res["boxes"]
scores: torch.Tensor = res["scores"]

# -------------------------------
# (1) 형태 필터: 최소 면적 & 종횡비
# -------------------------------
if boxes.numel() > 0:
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    w = (x2 - x1).clamp(min=0)
    h = (y2 - y1).clamp(min=0)
    area = w * h
    ar = torch.maximum(w / (h + 1e-6), h / (w + 1e-6))
    keep = (area >= MIN_AREA) & (ar <= MAX_AR)
    boxes = boxes[keep]
    scores = scores[keep]

# -------------------------------
# (2) 동적 스코어 컷: mean+alpha*std
# -------------------------------
if scores.numel() > 0:
    mean = scores.mean()
    std  = scores.std(unbiased=False)
    dyn_thresh = max(SCORE_THRESH, float(mean + DYN_ALPHA * std))
    keep = scores >= dyn_thresh
    boxes = boxes[keep]
    scores = scores[keep]

# -------------------------------
# (3) Containment/IoA 억제
#    IoU 말고 "거의 완전히 포함(큰 박스 안에 들어감)"도 제거
# -------------------------------
def containment_suppression(b, s, ioa_thr=0.9):
    # IoA: A∩B / A  (작은 박스 입장에서 대부분이 큰 박스에 포함되면 제거)
    if b.numel() == 0:
        return b, s
    idx = torch.argsort(s, descending=True)
    b = b[idx]
    s = s[idx]
    keep = []
    for i in range(b.size(0)):
        bi = b[i]
        x1i, y1i, x2i, y2i = bi
        wi = (x2i - x1i).clamp(min=0)
        hi = (y2i - y1i).clamp(min=0)
        areai = wi * hi + 1e-6
        contained = False
        for j in keep:
            bj = b[j]
            xx1 = torch.maximum(x1i, bj[0])
            yy1 = torch.maximum(y1i, bj[1])
            xx2 = torch.minimum(x2i, bj[2])
            yy2 = torch.minimum(y2i, bj[3])
            inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
            ioa = inter / areai  # A 기준
            if ioa > ioa_thr:
                contained = True
                break
        if not contained:
            keep.append(i)
    return b[keep], s[keep]

boxes, scores = containment_suppression(boxes, scores, ioa_thr=0.9)

# -------------------------------
# (4) Radius-NMS (센터 거리 기반 중복 억제)
# -------------------------------
def radius_nms(b, s, radius_px=RADIUS_NMS_PX):
    if b.numel() == 0:
        return b, s
    idx = torch.argsort(s, descending=True)
    b = b[idx]
    s = s[idx]
    keep = []
    centers = torch.stack([(b[:, 0] + b[:, 2]) * 0.5, (b[:, 1] + b[:, 3]) * 0.5], dim=1)
    for i in range(b.size(0)):
        ci = centers[i]
        drop = False
        for j in keep:
            cj = centers[j]
            dx = float(ci[0] - cj[0])
            dy = float(ci[1] - cj[1])
            if (dx * dx + dy * dy) <= (radius_px * radius_px):
                drop = True
                break
        if not drop:
            keep.append(i)
    return b[keep], s[keep]

boxes, scores = radius_nms(boxes, scores, radius_px=RADIUS_NMS_PX)

# -------------------------------
# (5) 최종 Top-K (원하면 사용)
# -------------------------------
if TOPK is not None and boxes.size(0) > TOPK:
    topk_idx = torch.topk(scores, k=TOPK, largest=True, sorted=True).indices
    boxes = boxes[topk_idx]
    scores = scores[topk_idx]

# -------------------------------
# 시각화
# -------------------------------
vis = cv2.cvtColor(cv2.imread(target_img_path), cv2.COLOR_BGR2RGB)
for box, score in zip(boxes, scores):
    x1, y1, x2, y2 = map(int, box.tolist())
    label = f"query {float(score):.2f}"
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)
    cv2.putText(vis, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2, lineType=cv2.LINE_AA)
    cv2.putText(vis, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 0, 0), 1, lineType=cv2.LINE_AA)

cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
print(f"✅ 최종 탐지 수: {boxes.size(0)}")
print(f"💾 결과 이미지 저장 위치: {output_path}")
