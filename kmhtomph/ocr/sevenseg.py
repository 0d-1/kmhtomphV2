from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import cv2

# Ordre standard des segments
SEG_ORDER = ["a", "b", "c", "d", "e", "f", "g"]

# Cartographie segments -> chiffre
DIGIT_MAP = {
    frozenset(["a", "b", "c", "d", "e", "f"]): 0,
    frozenset(["b", "c"]): 1,
    frozenset(["a", "b", "g", "e", "d"]): 2,
    frozenset(["a", "b", "c", "d", "g"]): 3,
    frozenset(["f", "g", "b", "c"]): 4,
    frozenset(["a", "f", "g", "c", "d"]): 5,
    frozenset(["a", "f", "g", "e", "c", "d"]): 6,
    frozenset(["a", "b", "c"]): 7,
    frozenset(["a", "b", "c", "d", "e", "f", "g"]): 8,
    frozenset(["a", "b", "c", "d", "f", "g"]): 9,
}


def _ensure_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _sevenseg_preprocess(gray: np.ndarray) -> np.ndarray:
    """
    Prétraitement robuste pour affichages 7-segments : upscale + tophat +
    seuillage OTSU, inversion si nécessaire, et fermetures pour reconnecter
    les barres horizontales (a/d/g).
    """
    g = _ensure_gray(gray)
    # upsample pour faciliter la morpho/threshold
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (5, 5), 0)

    # Top-hat pour supprimer le fond (éclairage non uniforme)
    kernel_tophat = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    tophat = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, kernel_tophat, iterations=1)

    # Binarisation (OTSU)
    thr = cv2.threshold(tophat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Assurer que segments = blanc
    if np.mean(thr) < 128:
        thr = cv2.bitwise_not(thr)

    # Reconnecter les segments horizontaux
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8), iterations=1)
    # Nettoyage léger
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
    return thr


def _split_digits_by_projection(bin_img: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Découpage par projection verticale pour obtenir les boîtes des digits."""
    h, w = bin_img.shape[:2]
    col_sum = (bin_img > 0).sum(axis=0).astype(np.float32)
    # Normaliser et seuiller faiblement
    if col_sum.max() > 0:
        col_sum /= col_sum.max()
    mask = col_sum > 0.1

    boxes: List[Tuple[int, int, int, int]] = []
    in_run = False
    start = 0
    for x in range(w):
        if mask[x] and not in_run:
            in_run = True
            start = x
        elif not mask[x] and in_run:
            in_run = False
            end = x
            if end - start >= max(4, int(0.03 * w)):
                boxes.append((start, 0, end - start, h))
    if in_run:
        end = w - 1
        if end - start >= max(4, int(0.03 * w)):
            boxes.append((start, 0, end - start, h))
    return boxes


def _classify_7seg_digit(patch_bin: np.ndarray) -> Tuple[Optional[int], float]:
    """Classification d’un digit 7-segments + score de confiance (0..1)."""
    h, w = patch_bin.shape[:2]
    # Ecarter le bruit minuscule
    if h < 8 or w < 5:
        return None, 0.0

    # Normaliser
    patch = cv2.resize(patch_bin, (32, 48), interpolation=cv2.INTER_NEAREST)
    H, W = patch.shape[:2]

    # Définition des 7 zones (approximatives mais robustes)
    def zone(y0, y1, x0, x1):
        y0 = max(0, min(H - 1, int(y0)))
        y1 = max(0, min(H, int(y1)))
        x0 = max(0, min(W - 1, int(x0)))
        x1 = max(0, min(W, int(x1)))
        z = patch[y0:y1, x0:x1]
        return (z > 0).mean() if z.size else 0.0

    a = zone(0.02 * H, 0.18 * H, 0.2 * W, 0.8 * W)
    d = zone(0.82 * H, 0.98 * H, 0.2 * W, 0.8 * W)
    g = zone(0.48 * H, 0.60 * H, 0.2 * W, 0.8 * W)
    f = zone(0.18 * H, 0.50 * H, 0.05 * W, 0.25 * W)
    e = zone(0.50 * H, 0.82 * H, 0.05 * W, 0.25 * W)
    b = zone(0.18 * H, 0.50 * H, 0.75 * W, 0.95 * W)
    c = zone(0.50 * H, 0.82 * H, 0.75 * W, 0.95 * W)

    seg_vals = dict(zip(SEG_ORDER, [a, b, c, d, e, f, g]))

    best_digit: Optional[int] = None
    best_score = -1.0
    for segs, digit in DIGIT_MAP.items():
        score = 0.0
        for name in SEG_ORDER:
            target = 1.0 if name in segs else 0.0
            val = float(seg_vals[name])
            score += max(0.0, 1.0 - abs(val - target))
        score /= float(len(SEG_ORDER))
        if score > best_score:
            best_digit = digit
            best_score = score

    if best_score < 0.55:
        return None, float(best_score)

    return best_digit, float(min(1.0, max(0.0, best_score)))


def sevenseg_ocr(bgr_or_gray: np.ndarray) -> Tuple[Optional[str], float, np.ndarray]:
    """
    OCR 7-segments « maison ».
    Retourne (texte, confiance, image_debug_bgr).
    """
    gray = _ensure_gray(bgr_or_gray)
    bin_img = _sevenseg_preprocess(gray)

    boxes = _split_digits_by_projection(bin_img)
    if not boxes:
        return None, 0.0, cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    boxes.sort(key=lambda b: b[0])

    digits: List[int] = []
    digit_scores: List[float] = []
    for (x, y, w, h) in boxes:
        patch = bin_img[y:y + h, x:x + w]
        d, score = _classify_7seg_digit(patch)
        if d is None:
            return None, 0.0, cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
        digits.append(d)
        digit_scores.append(score)

    txt = "".join(str(d) for d in digits)
    conf = float(np.mean(digit_scores)) if digit_scores else 0.0
    conf = float(min(1.0, max(0.0, conf)))

    dbg = cv2.cvtColor(bin_img, cv2.COLOR_GRAY2BGR)
    for (x, y, w, h) in boxes:
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 0), 1)

    return txt, float(conf), dbg
