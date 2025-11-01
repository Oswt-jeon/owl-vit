import argparse
import math
import os

import cv2
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor


# ------------------------------------------------------------
# 모델 설정
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "google/owlv2-large-patch14-ensemble"
processor = Owlv2Processor.from_pretrained(model_id)
model = Owlv2ForObjectDetection.from_pretrained(model_id).to(device).eval()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OWL-ViT 이미지 유사 객체 탐지")
    parser.add_argument("--target", default="images/target4.jpeg", help="탐지 대상 이미지 경로")
    parser.add_argument("--query", default="images/wang.jpg", help="쿼리 이미지 경로")
    parser.add_argument("--output", default="images/result.jpg", help="결과 이미지 저장 경로")
    parser.add_argument("--score-thresh", type=float, default=0.45, help="1차 점수 임계값")
    parser.add_argument("--dyn-alpha", type=float, default=0.5, help="동적 점수 컷: mean + alpha*std")
    parser.add_argument("--min-area-ratio", type=float, default=0.0015, help="박스 최소 면적 비율")
    parser.add_argument("--max-ar", type=float, default=6.0, help="허용 최대 종횡비")
    parser.add_argument("--nms-thresh", type=float, default=0.35, help="post_process NMS 임계값")
    parser.add_argument("--radius-nms", type=float, default=32.0, help="중심거리 기반 억제 픽셀 반경")
    parser.add_argument("--topk", type=int, default=None, help="최종 상위 K개 박스만 유지")
    parser.add_argument(
        "--scales",
        default="1.0,1.3",
        help="콤마로 구분된 스케일 팩터 목록 (예: 1.0,1.2,1.5)",
    )
    parser.add_argument(
        "--base-long-side",
        type=int,
        default=None,
        help="타겟 이미지의 긴 변을 이 값으로 리사이즈한 뒤 스케일 적용",
    )
    parser.add_argument("--skip-dyn", action="store_true", help="동적 점수 컷 비활성화")
    parser.add_argument("--verbose", action="store_true", help="중간 점수 정보 출력")
    return parser.parse_args()


def parse_scales(scale_text: str) -> list[float]:
    scales: list[float] = []
    for chunk in scale_text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            value = float(chunk)
        except ValueError as exc:
            raise ValueError(f"'{chunk}'를 float으로 변환할 수 없습니다.") from exc
        if value <= 0:
            raise ValueError(f"스케일 값은 양수여야 합니다: {value}")
        scales.append(value)
    if not scales:
        scales = [1.0]
    return sorted(set(scales))


def containment_suppression(boxes: torch.Tensor, scores: torch.Tensor, ioa_thr: float = 0.9):
    if boxes.numel() == 0:
        return boxes, scores
    idx = torch.argsort(scores, descending=True)
    boxes = boxes[idx]
    scores = scores[idx]
    keep: list[int] = []
    for i in range(boxes.size(0)):
        bi = boxes[i]
        x1i, y1i, x2i, y2i = bi
        wi = (x2i - x1i).clamp(min=0)
        hi = (y2i - y1i).clamp(min=0)
        areai = wi * hi + 1e-6
        contained = False
        for j in keep:
            bj = boxes[j]
            xx1 = torch.maximum(x1i, bj[0])
            yy1 = torch.maximum(y1i, bj[1])
            xx2 = torch.minimum(x2i, bj[2])
            yy2 = torch.minimum(y2i, bj[3])
            inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
            ioa = inter / areai
            if ioa > ioa_thr:
                contained = True
                break
        if not contained:
            keep.append(i)
    return boxes[keep], scores[keep]


def radius_nms(boxes: torch.Tensor, scores: torch.Tensor, radius_px: float = 48.0):
    if boxes.numel() == 0:
        return boxes, scores
    idx = torch.argsort(scores, descending=True)
    boxes = boxes[idx]
    scores = scores[idx]
    keep: list[int] = []
    centers = torch.stack([(boxes[:, 0] + boxes[:, 2]) * 0.5, (boxes[:, 1] + boxes[:, 3]) * 0.5], dim=1)
    radius_sq = radius_px * radius_px
    for i in range(boxes.size(0)):
        ci = centers[i]
        drop = False
        for j in keep:
            cj = centers[j]
            dx = float(ci[0] - cj[0])
            dy = float(ci[1] - cj[1])
            if (dx * dx + dy * dy) <= radius_sq:
                drop = True
                break
        if not drop:
            keep.append(i)
    return boxes[keep], scores[keep]


def run_single_scale(
    target_img: Image.Image,
    query_img: Image.Image,
    score_thresh: float,
    nms_thresh: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = processor(images=target_img, query_images=[query_img], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.image_guided_detection(**inputs)
    post = processor.post_process_image_guided_detection(
        outputs=outputs,
        target_sizes=torch.tensor([(target_img.height, target_img.width)], device=device),
        threshold=score_thresh,
        nms_threshold=nms_thresh,
    )[0]
    boxes = post["boxes"].to("cpu")
    scores = post["scores"].to("cpu")
    return boxes, scores


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.target):
        raise FileNotFoundError(f"❌ target 이미지가 존재하지 않습니다: {args.target}")
    if not os.path.exists(args.query):
        raise FileNotFoundError(f"❌ query 이미지가 존재하지 않습니다: {args.query}")

    try:
        scales = parse_scales(args.scales)
    except ValueError as err:
        raise SystemExit(f"--scales 파싱 실패: {err}") from err

    target_img = Image.open(args.target).convert("RGB")
    query_img = Image.open(args.query).convert("RGB")
    orig_width, orig_height = target_img.width, target_img.height
    img_area = float(orig_width * orig_height)
    min_area = img_area * args.min_area_ratio

    base_factor = 1.0
    if args.base_long_side:
        long_side = max(orig_width, orig_height)
        base_factor = max(args.base_long_side / long_side, 1e-6)

    collected_boxes: list[torch.Tensor] = []
    collected_scores: list[torch.Tensor] = []
    for scale in scales:
        total_factor = base_factor * scale
        if not math.isclose(total_factor, 1.0, rel_tol=1e-3):
            new_w = max(1, int(round(orig_width * total_factor)))
            new_h = max(1, int(round(orig_height * total_factor)))
            resized_target = target_img.resize((new_w, new_h), resample=Image.BICUBIC)
        else:
            resized_target = target_img
        boxes, scores = run_single_scale(resized_target, query_img, args.score_thresh, args.nms_thresh)
        if boxes.numel() == 0:
            continue
        if not math.isclose(total_factor, 1.0, rel_tol=1e-3):
            boxes = boxes / total_factor
        collected_boxes.append(boxes)
        collected_scores.append(scores)
        if args.verbose:
            score_samples = ", ".join(f"{float(s):.3f}" for s in scores[:5])
            print(f"[scale {total_factor:.2f}] raw detections: {boxes.size(0)} | scores: {score_samples}")

    if collected_boxes:
        boxes = torch.cat(collected_boxes, dim=0)
        scores = torch.cat(collected_scores, dim=0)
    else:
        boxes = torch.empty((0, 4), dtype=torch.float32)
        scores = torch.empty((0,), dtype=torch.float32)

    # -------------------------------
    # (1) 형태 필터
    # -------------------------------
    if boxes.numel() > 0:
        x1, y1, x2, y2 = boxes.unbind(dim=1)
        w = (x2 - x1).clamp(min=0)
        h = (y2 - y1).clamp(min=0)
        area = w * h
        aspect_ratio = torch.maximum(w / (h + 1e-6), h / (w + 1e-6))
        keep = (area >= min_area) & (aspect_ratio <= args.max_ar)
        boxes = boxes[keep]
        scores = scores[keep]

    # -------------------------------
    # (2) 동적 스코어 컷
    # -------------------------------
    if scores.numel() > 0 and not args.skip_dyn and args.dyn_alpha >= 0:
        mean = scores.mean()
        std = scores.std(unbiased=False)
        dyn_thresh = max(args.score_thresh, float(mean + args.dyn_alpha * std))
        dyn_thresh = min(dyn_thresh, float(scores.max()))
        keep = scores >= dyn_thresh
        if keep.sum() == 0:
            keep = scores >= scores.max()
        boxes = boxes[keep]
        scores = scores[keep]

    # -------------------------------
    # (3) Containment/IoA 억제
    # -------------------------------
    boxes, scores = containment_suppression(boxes, scores, ioa_thr=0.9)

    # -------------------------------
    # (4) Radius-NMS
    # -------------------------------
    boxes, scores = radius_nms(boxes, scores, radius_px=args.radius_nms)

    # -------------------------------
    # (5) 최종 Top-K
    # -------------------------------
    if args.topk and args.topk > 0 and boxes.size(0) > args.topk:
        topk_idx = torch.topk(scores, k=args.topk, largest=True, sorted=True).indices
        boxes = boxes[topk_idx]
        scores = scores[topk_idx]

    vis = cv2.cvtColor(cv2.imread(args.target), cv2.COLOR_BGR2RGB)
    vis_h, vis_w = vis.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_scale = 0.8
    text_thickness = 2
    text_pad = 4

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box.tolist())
        label = f"query {float(score):.3f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 3)

        (text_w, text_h), baseline = cv2.getTextSize(label, font, text_scale, text_thickness)
        baseline = int(math.ceil(baseline))

        text_x = x1 + text_pad
        text_y = y1 + text_pad + text_h
        if text_x + text_w + text_pad > x2:
            text_x = min(max(x2 - text_w - text_pad, text_pad), vis_w - text_w - text_pad)
        if text_y + baseline + text_pad > y2:
            text_y = max(y2 - text_pad - baseline, text_pad + text_h)

        x_upper = max(text_pad, vis_w - text_w - text_pad)
        y_upper = max(text_pad + text_h, vis_h - text_pad)
        text_x = min(max(text_x, text_pad), x_upper)
        text_y = min(max(text_y, text_pad + text_h), y_upper)

        bg_left = max(text_x - text_pad, 0)
        bg_top = max(text_y - text_h - baseline - text_pad, 0)
        bg_right = min(text_x + text_w + text_pad, vis_w)
        bg_bottom = min(text_y + baseline + text_pad, vis_h)
        cv2.rectangle(vis, (bg_left, bg_top), (bg_right, bg_bottom), (0, 0, 0), thickness=-1)

        cv2.putText(
            vis,
            label,
            (text_x, text_y),
            font,
            text_scale,
            (255, 255, 255),
            text_thickness,
            lineType=cv2.LINE_AA,
        )
        cv2.putText(
            vis,
            label,
            (text_x, text_y),
            font,
            text_scale,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )

    cv2.imwrite(args.output, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    if boxes.size(0) == 0:
        print("⚠️ 최종 탐지가 없습니다. --scales, --score-thresh, --dyn-alpha 등을 조정해 보세요.")
    else:
        summary_scores = ", ".join(f"{float(s):.3f}" for s in scores.tolist())
        print(f"✅ 최종 탐지 수: {boxes.size(0)} | scores: {summary_scores}")
    print(f"💾 결과 이미지 저장 위치: {args.output}")


if __name__ == "__main__":
    main()
