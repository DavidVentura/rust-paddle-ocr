//! Postprocessing Utilities
//!
//! Provides post-processing functions for text detection results, including bounding box extraction, NMS, box merging, etc.

use image::GrayImage;
use imageproc::contours::{find_contours, Contour};
use imageproc::point::Point;
use imageproc::rect::Rect;

/// Text bounding box
#[derive(Debug, Clone)]
pub struct TextBox {
    /// Bounding box rectangle
    pub rect: Rect,
    /// Confidence score
    pub score: f32,
    /// Four corner points (optional, for rotated boxes)
    pub points: Option<[Point<f32>; 4]>,
    /// Full DBNet contour in original-image coords (optional). Carries shape info that the
    /// 4-corner reduction throws away — e.g. arched/curved text lines on cylindrical labels.
    pub contour: Option<Vec<Point<f32>>>,
}

impl TextBox {
    /// Create new text bounding box
    pub fn new(rect: Rect, score: f32) -> Self {
        Self {
            rect,
            score,
            points: None,
            contour: None,
        }
    }

    /// Create with corner points
    pub fn with_points(rect: Rect, score: f32, points: [Point<f32>; 4]) -> Self {
        Self {
            rect,
            score,
            points: Some(points),
            contour: None,
        }
    }

    /// Create with corner points and full contour
    pub fn with_points_and_contour(
        rect: Rect,
        score: f32,
        points: [Point<f32>; 4],
        contour: Vec<Point<f32>>,
    ) -> Self {
        Self {
            rect,
            score,
            points: Some(points),
            contour: Some(contour),
        }
    }

    /// Calculate area
    pub fn area(&self) -> u32 {
        self.rect.width() * self.rect.height()
    }

    /// Expand bounding box
    pub fn expand(&self, border: u32, max_width: u32, max_height: u32) -> Self {
        let x = (self.rect.left() - border as i32).max(0) as u32;
        let y = (self.rect.top() - border as i32).max(0) as u32;
        let right = ((self.rect.left() as u32 + self.rect.width()) + border).min(max_width);
        let bottom = ((self.rect.top() as u32 + self.rect.height()) + border).min(max_height);

        // 确保 right >= x 和 bottom >= y，避免减法溢出
        let width = if right > x { right - x } else { 1 };
        let height = if bottom > y { bottom - y } else { 1 };

        Self {
            rect: Rect::at(x as i32, y as i32).of_size(width, height),
            score: self.score,
            points: self.points,
            contour: self.contour.clone(),
        }
    }
}

/// Extract text bounding boxes from segmentation mask
///
/// # Parameters
/// - `mask`: Binarized mask (0 or 255)
/// - `width`: Mask width
/// - `height`: Mask height
/// - `original_width`: Original image width
/// - `original_height`: Original image height
/// - `min_area`: Minimum bounding box area
/// - `box_threshold`: Bounding box score threshold
pub fn extract_boxes_from_mask(
    mask: &[u8],
    width: u32,
    height: u32,
    original_width: u32,
    original_height: u32,
    min_area: u32,
    _box_threshold: f32,
) -> Vec<TextBox> {
    extract_boxes_from_mask_with_padding(
        mask,
        width,
        height,
        width,
        height,
        original_width,
        original_height,
        min_area,
        _box_threshold,
    )
}

/// Extract text bounding boxes from segmentation mask with padding
///
/// # Parameters
/// - `mask`: Binarized mask (0 or 255)
/// - `mask_width`: Mask width (including padding)
/// - `mask_height`: Mask height (including padding)
/// - `valid_width`: Valid region width (excluding padding)
/// - `valid_height`: Valid region height (excluding padding)
/// - `original_width`: Original image width
/// - `original_height`: Original image height
/// - `min_area`: Minimum bounding box area
/// - `box_threshold`: Bounding box score threshold
pub fn extract_boxes_from_mask_with_padding(
    mask: &[u8],
    mask_width: u32,
    mask_height: u32,
    valid_width: u32,
    valid_height: u32,
    original_width: u32,
    original_height: u32,
    min_area: u32,
    _box_threshold: f32,
) -> Vec<TextBox> {
    extract_boxes_with_unclip(
        mask,
        mask_width,
        mask_height,
        valid_width,
        valid_height,
        original_width,
        original_height,
        min_area,
        1.5, // 默认 unclip_ratio
    )
}

/// Extract connected-component boxes from a binarized segmentation mask without DB unclip.
///
/// This is useful for experiments that want tighter pre-unclip regions, which can
/// sometimes approximate word-level boxes better than the default line-oriented DB output.
pub fn extract_boxes_from_mask_without_unclip(
    mask: &[u8],
    mask_width: u32,
    mask_height: u32,
    valid_width: u32,
    valid_height: u32,
    original_width: u32,
    original_height: u32,
    min_area: u32,
) -> Vec<TextBox> {
    extract_boxes_from_mask_with_optional_unclip(
        mask,
        mask_width,
        mask_height,
        valid_width,
        valid_height,
        original_width,
        original_height,
        min_area,
        None,
    )
}

/// Extract text bounding boxes from segmentation mask (with unclip expansion)
///
/// Core of DB algorithm is to perform unclip expansion on detected contours,
/// because model output segmentation mask is usually smaller than actual text region.
pub fn extract_boxes_with_unclip(
    mask: &[u8],
    mask_width: u32,
    mask_height: u32,
    valid_width: u32,
    valid_height: u32,
    original_width: u32,
    original_height: u32,
    min_area: u32,
    unclip_ratio: f32,
) -> Vec<TextBox> {
    extract_boxes_from_mask_with_optional_unclip(
        mask,
        mask_width,
        mask_height,
        valid_width,
        valid_height,
        original_width,
        original_height,
        min_area,
        Some(unclip_ratio),
    )
}

fn extract_boxes_from_mask_with_optional_unclip(
    mask: &[u8],
    mask_width: u32,
    mask_height: u32,
    valid_width: u32,
    valid_height: u32,
    original_width: u32,
    original_height: u32,
    min_area: u32,
    unclip_ratio: Option<f32>,
) -> Vec<TextBox> {
    // Create grayscale image
    let gray_image = GrayImage::from_raw(mask_width, mask_height, mask.to_vec())
        .unwrap_or_else(|| GrayImage::new(mask_width, mask_height));

    // Find contours
    let contours = find_contours::<i32>(&gray_image);

    // Calculate scale ratio (from valid region to original image)
    let scale_x = original_width as f32 / valid_width as f32;
    let scale_y = original_height as f32 / valid_height as f32;

    let mut boxes = Vec::new();

    for contour in contours {
        // Only keep outer contours (without parent), filter out inner/nested contours
        // This avoids producing overlapping detection boxes
        if contour.parent.is_some() {
            continue;
        }

        if contour.points.len() < 4 {
            continue;
        }

        // Calculate bounding box
        let (min_x, min_y, max_x, max_y) = get_contour_bounds(&contour);

        // Filter out contours in padding area
        if min_x >= valid_width as i32 || min_y >= valid_height as i32 {
            continue;
        }

        // Clip to valid region
        let min_x = min_x.max(0);
        let min_y = min_y.max(0);
        let max_x = max_x.min(valid_width as i32);
        let max_y = max_y.min(valid_height as i32);

        let box_width = (max_x - min_x) as u32;
        let box_height = (max_y - min_y) as u32;

        // Filter boxes that are too small
        if box_width * box_height < min_area {
            continue;
        }

        let (expanded_min_x, expanded_min_y, expanded_max_x, expanded_max_y) =
            if let Some(unclip_ratio) = unclip_ratio {
                // DB algorithm uses area and perimeter to calculate expansion distance:
                // distance = Area * unclip_ratio / Perimeter
                let area = box_width as f32 * box_height as f32;
                let perimeter = 2.0 * (box_width + box_height) as f32;
                let expand_dist = (area * unclip_ratio / perimeter).max(1.0);

                (
                    (min_x as f32 - expand_dist).max(0.0) as i32,
                    (min_y as f32 - expand_dist).max(0.0) as i32,
                    (max_x as f32 + expand_dist).min(valid_width as f32) as i32,
                    (max_y as f32 + expand_dist).min(valid_height as f32) as i32,
                )
            } else {
                (min_x, min_y, max_x, max_y)
            };

        let expanded_w = (expanded_max_x - expanded_min_x) as u32;
        let expanded_h = (expanded_max_y - expanded_min_y) as u32;

        // Scale to original image size
        let scaled_x = (expanded_min_x as f32 * scale_x) as i32;
        let scaled_y = (expanded_min_y as f32 * scale_y) as i32;
        let scaled_w = (expanded_w as f32 * scale_x) as u32;
        let scaled_h = (expanded_h as f32 * scale_y) as u32;

        // Ensure boundaries are within valid range
        let final_x = scaled_x.max(0) as u32;
        let final_y = scaled_y.max(0) as u32;
        let final_w = scaled_w.min(original_width.saturating_sub(final_x));
        let final_h = scaled_h.min(original_height.saturating_sub(final_y));

        if final_w > 0 && final_h > 0 {
            let rect = Rect::at(final_x as i32, final_y as i32).of_size(final_w, final_h);
            // Compute oriented bounding box from the contour points (via PCA → principal
            // axis), scaled to original-image coords. Captures the text-region rotation that
            // the AABB throws away.
            let oriented = compute_oriented_rect_from_contour(
                &contour.points,
                scale_x,
                scale_y,
            );
            // Also keep the raw contour in original-image coords. Lets downstream code see
            // arched/curved text shapes that the 4-corner reduction throws away.
            let scaled_contour: Vec<Point<f32>> = contour
                .points
                .iter()
                .map(|p| Point::new(p.x as f32 * scale_x, p.y as f32 * scale_y))
                .collect();
            boxes.push(match oriented {
                Some(points) => TextBox::with_points_and_contour(rect, 1.0, points, scaled_contour),
                None => TextBox::new(rect, 1.0),
            });
        }
    }

    boxes
}

/// Compute the oriented bounding rectangle of [contour_points] (in mask coordinates) and
/// return its 4 corners in original-image coords (scaled by [scale_x], [scale_y]).
///
/// Uses PCA on the contour points: the principal eigenvector of the 2D covariance matrix is
/// the long axis of the text region. The 4 corners are obtained by projecting all contour
/// points onto the principal axis (u) and the perpendicular (v), taking the min/max of each,
/// and transforming back to image coords.
///
/// Returns corners ordered as a clockwise traversal starting from the top-left in the
/// oriented frame (uMin, vMin), (uMax, vMin), (uMax, vMax), (uMin, vMax).
fn compute_oriented_rect_from_contour(
    contour_points: &[Point<i32>],
    scale_x: f32,
    scale_y: f32,
) -> Option<[Point<f32>; 4]> {
    if contour_points.len() < 4 {
        return None;
    }
    let n = contour_points.len() as f32;
    let mut mean_x = 0.0f32;
    let mut mean_y = 0.0f32;
    for p in contour_points {
        mean_x += p.x as f32;
        mean_y += p.y as f32;
    }
    mean_x /= n;
    mean_y /= n;

    let mut cxx = 0.0f32;
    let mut cyy = 0.0f32;
    let mut cxy = 0.0f32;
    for p in contour_points {
        let dx = p.x as f32 - mean_x;
        let dy = p.y as f32 - mean_y;
        cxx += dx * dx;
        cyy += dy * dy;
        cxy += dx * dy;
    }
    cxx /= n;
    cyy /= n;
    cxy /= n;

    // 2x2 symmetric eigendecomposition. Larger eigenvalue's eigenvector = principal axis.
    let trace = cxx + cyy;
    let det = cxx * cyy - cxy * cxy;
    let disc = (trace * trace - 4.0 * det).max(0.0).sqrt();
    let lambda1 = (trace + disc) * 0.5;

    let (ex, ey) = if cxy.abs() > 1e-6 {
        (lambda1 - cyy, cxy)
    } else if cxx >= cyy {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let norm = (ex * ex + ey * ey).sqrt().max(1e-6);
    let ux = ex / norm;
    let uy = ey / norm;
    // Perpendicular axis (rotated +90° from u).
    let vx = -uy;
    let vy = ux;

    let mut u_min = f32::INFINITY;
    let mut u_max = f32::NEG_INFINITY;
    let mut v_min = f32::INFINITY;
    let mut v_max = f32::NEG_INFINITY;
    for p in contour_points {
        let dx = p.x as f32 - mean_x;
        let dy = p.y as f32 - mean_y;
        let u = dx * ux + dy * uy;
        let v = dx * vx + dy * vy;
        if u < u_min { u_min = u; }
        if u > u_max { u_max = u; }
        if v < v_min { v_min = v; }
        if v > v_max { v_max = v; }
    }

    let back = |u: f32, v: f32| -> Point<f32> {
        Point::new(
            (mean_x + u * ux + v * vx) * scale_x,
            (mean_y + u * uy + v * vy) * scale_y,
        )
    };
    // Ensure the principal axis points right-ish in image space so the ordering is intuitive
    // (TL, TR, BR, BL).
    let (u_lo, u_hi, v_lo, v_hi) = if ux >= 0.0 {
        (u_min, u_max, v_min, v_max)
    } else {
        (u_max, u_min, v_max, v_min)
    };
    Some([
        back(u_lo, v_lo),
        back(u_hi, v_lo),
        back(u_hi, v_hi),
        back(u_lo, v_hi),
    ])
}

/// Get contour bounds
fn get_contour_bounds(contour: &Contour<i32>) -> (i32, i32, i32, i32) {
    let mut min_x = i32::MAX;
    let mut min_y = i32::MAX;
    let mut max_x = i32::MIN;
    let mut max_y = i32::MIN;

    for point in &contour.points {
        min_x = min_x.min(point.x);
        min_y = min_y.min(point.y);
        max_x = max_x.max(point.x);
        max_y = max_y.max(point.y);
    }

    (min_x, min_y, max_x, max_y)
}

/// Calculate containment ratio of one box inside another
fn compute_containment_ratio(inner: &Rect, outer: &Rect) -> f32 {
    let x1 = inner.left().max(outer.left());
    let y1 = inner.top().max(outer.top());
    let x2 = (inner.left() + inner.width() as i32).min(outer.left() + outer.width() as i32);
    let y2 = (inner.top() + inner.height() as i32).min(outer.top() + outer.height() as i32);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) as f32 * (y2 - y1) as f32;
    let inner_area = inner.width() as f32 * inner.height() as f32;

    if inner_area <= 0.0 {
        0.0
    } else {
        intersection / inner_area
    }
}

/// Non-Maximum Suppression (NMS)
///
/// Filter overlapping bounding boxes, keep ones with highest scores
/// Also filters small boxes that are largely contained within other boxes
///
/// # Parameters
/// - `boxes`: List of bounding boxes
/// - `iou_threshold`: IoU threshold, boxes exceeding this value are considered overlapping
pub fn nms(boxes: &[TextBox], iou_threshold: f32) -> Vec<TextBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // Sort by score descending, area descending (keep boxes with higher score and larger area first)
    let mut indices: Vec<usize> = (0..boxes.len()).collect();
    indices.sort_by(|&a, &b| {
        // First sort by score descending
        let score_cmp = boxes[b]
            .score
            .partial_cmp(&boxes[a].score)
            .unwrap_or(std::cmp::Ordering::Equal);
        if score_cmp != std::cmp::Ordering::Equal {
            return score_cmp;
        }
        // When scores are equal, sort by area descending (prefer larger boxes)
        boxes[b].area().cmp(&boxes[a].area())
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; boxes.len()];

    for (pos, &i) in indices.iter().enumerate() {
        if suppressed[i] {
            continue;
        }

        keep.push(boxes[i].clone());

        // Check all subsequent boxes (lower score or smaller area)
        for &j in indices.iter().skip(pos + 1) {
            if suppressed[j] {
                continue;
            }

            // Check IoU
            let iou = compute_iou(&boxes[i].rect, &boxes[j].rect);
            if iou > iou_threshold {
                suppressed[j] = true;
                continue;
            }

            // Check containment relationship: if j is largely contained (>50%) by i, suppress j
            let containment_j_in_i = compute_containment_ratio(&boxes[j].rect, &boxes[i].rect);
            if containment_j_in_i > 0.5 {
                suppressed[j] = true;
                continue;
            }

            // Check reverse containment: if i is largely contained (>70%) by j,
            // since i was selected first (higher score or larger area), suppress j
            let containment_i_in_j = compute_containment_ratio(&boxes[i].rect, &boxes[j].rect);
            if containment_i_in_j > 0.7 {
                suppressed[j] = true;
                continue;
            }
        }
    }

    keep
}

/// Calculate IoU (Intersection over Union) of two rectangles
pub fn compute_iou(a: &Rect, b: &Rect) -> f32 {
    let x1 = a.left().max(b.left());
    let y1 = a.top().max(b.top());
    let x2 = (a.left() + a.width() as i32).min(b.left() + b.width() as i32);
    let y2 = (a.top() + a.height() as i32).min(b.top() + b.height() as i32);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) as f32 * (y2 - y1) as f32;
    let area_a = a.width() as f32 * a.height() as f32;
    let area_b = b.width() as f32 * b.height() as f32;
    let union = area_a + area_b - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

/// Merge adjacent bounding boxes
///
/// Merge bounding boxes that are close to each other into one
///
/// # Parameters
/// - `boxes`: List of bounding boxes
/// - `distance_threshold`: Distance threshold, boxes below this value will be merged
pub fn merge_adjacent_boxes(boxes: &[TextBox], distance_threshold: i32) -> Vec<TextBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut merged = Vec::new();
    let mut used = vec![false; boxes.len()];

    for i in 0..boxes.len() {
        if used[i] {
            continue;
        }

        let mut current = boxes[i].rect;
        let mut group_score = boxes[i].score;
        let mut count = 1;
        used[i] = true;

        // Find boxes that can be merged
        loop {
            let mut found = false;

            for j in 0..boxes.len() {
                if used[j] {
                    continue;
                }

                if can_merge(&current, &boxes[j].rect, distance_threshold) {
                    current = merge_rects(&current, &boxes[j].rect);
                    group_score += boxes[j].score;
                    count += 1;
                    used[j] = true;
                    found = true;
                }
            }

            if !found {
                break;
            }
        }

        merged.push(TextBox::new(current, group_score / count as f32));
    }

    merged
}

/// Check if two boxes can be merged
fn can_merge(a: &Rect, b: &Rect, threshold: i32) -> bool {
    // Calculate vertical distance
    let a_bottom = a.top() + a.height() as i32;
    let b_bottom = b.top() + b.height() as i32;

    let _vertical_dist = if a.top() > b_bottom {
        a.top() - b_bottom
    } else if b.top() > a_bottom {
        b.top() - a_bottom
    } else {
        0 // Vertical overlap
    };

    // Calculate horizontal distance
    let a_right = a.left() + a.width() as i32;
    let b_right = b.left() + b.width() as i32;

    let horizontal_dist = if a.left() > b_right {
        a.left() - b_right
    } else if b.left() > a_right {
        b.left() - a_right
    } else {
        0 // Horizontal overlap
    };

    // Check if on same line (vertical overlap) and horizontal distance is less than threshold
    let vertical_overlap = !(a.top() > b_bottom || b.top() > a_bottom);

    vertical_overlap && horizontal_dist <= threshold
}

/// Merge two rectangles
fn merge_rects(a: &Rect, b: &Rect) -> Rect {
    let x1 = a.left().min(b.left());
    let y1 = a.top().min(b.top());
    let x2 = (a.left() + a.width() as i32).max(b.left() + b.width() as i32);
    let y2 = (a.top() + a.height() as i32).max(b.top() + b.height() as i32);

    Rect::at(x1, y1).of_size((x2 - x1) as u32, (y2 - y1) as u32)
}

/// Sort bounding boxes by reading order (top to bottom, left to right)
pub fn sort_boxes_by_reading_order(boxes: &mut [TextBox]) {
    boxes.sort_by(|a, b| {
        // First sort by y coordinate (row)
        let y_cmp = a.rect.top().cmp(&b.rect.top());
        if y_cmp != std::cmp::Ordering::Equal {
            return y_cmp;
        }
        // Same row, sort by x coordinate
        a.rect.left().cmp(&b.rect.left())
    });
}

/// Group bounding boxes by line
///
/// Group boxes with close y coordinates into the same line
pub fn group_boxes_by_line(boxes: &[TextBox], line_threshold: i32) -> Vec<Vec<TextBox>> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut sorted_boxes = boxes.to_vec();
    sorted_boxes.sort_by_key(|b| b.rect.top());

    let mut lines: Vec<Vec<TextBox>> = Vec::new();
    let mut current_line: Vec<TextBox> = vec![sorted_boxes[0].clone()];
    let mut current_y = sorted_boxes[0].rect.top();

    for box_item in sorted_boxes.iter().skip(1) {
        if (box_item.rect.top() - current_y).abs() <= line_threshold {
            current_line.push(box_item.clone());
        } else {
            // Sort current line by x
            current_line.sort_by_key(|b| b.rect.left());
            lines.push(current_line);
            current_line = vec![box_item.clone()];
            current_y = box_item.rect.top();
        }
    }

    // Add last line
    if !current_line.is_empty() {
        current_line.sort_by_key(|b| b.rect.left());
        lines.push(current_line);
    }

    lines
}

/// Merge bounding boxes from multiple detection results (for high precision mode)
///
/// # Parameters
/// - `results`: Multiple detection results, each element is (boxes, offset_x, offset_y, scale)
/// - `iou_threshold`: NMS IoU threshold
pub fn merge_multi_scale_results(
    results: &[(Vec<TextBox>, u32, u32, f32)],
    iou_threshold: f32,
) -> Vec<TextBox> {
    let mut all_boxes = Vec::new();

    for (boxes, offset_x, offset_y, scale) in results {
        for box_item in boxes {
            // Convert box coordinates to original image coordinate system
            let scaled_x = (box_item.rect.left() as f32 / scale) as i32 + *offset_x as i32;
            let scaled_y = (box_item.rect.top() as f32 / scale) as i32 + *offset_y as i32;
            let scaled_w = (box_item.rect.width() as f32 / scale) as u32;
            let scaled_h = (box_item.rect.height() as f32 / scale) as u32;

            let rect = Rect::at(scaled_x, scaled_y).of_size(scaled_w, scaled_h);
            all_boxes.push(TextBox::new(rect, box_item.score));
        }
    }

    // Apply NMS to remove duplicates
    nms(&all_boxes, iou_threshold)
}

// ============== Traditional Algorithm Detection ==============

/// Detect text regions using traditional algorithm (suitable for solid background)
///
/// Based on OTSU binarization + connected component analysis, suitable for:
/// - Document images with solid background
/// - High contrast text
/// - As supplement to deep learning detection
///
/// # Parameters
/// - `gray_image`: Grayscale image
/// - `min_area`: Minimum text region area
/// - `expand_ratio`: Bounding box expansion ratio
pub fn detect_text_traditional(
    gray_image: &GrayImage,
    min_area: u32,
    expand_ratio: f32,
) -> Vec<TextBox> {
    let (width, height) = gray_image.dimensions();

    // 1. Calculate OTSU threshold
    let threshold = otsu_threshold(gray_image);

    // 2. Binarization
    let binary: Vec<u8> = gray_image
        .pixels()
        .map(|p| if p.0[0] < threshold { 255 } else { 0 })
        .collect();

    // 3. Create binary image and find contours
    let binary_image =
        GrayImage::from_raw(width, height, binary).unwrap_or_else(|| GrayImage::new(width, height));
    let contours = find_contours::<i32>(&binary_image);

    // 4. Extract bounding boxes
    let mut boxes = Vec::new();
    for contour in contours {
        if contour.points.len() < 4 {
            continue;
        }

        let (min_x, min_y, max_x, max_y) = get_contour_bounds(&contour);
        let box_width = (max_x - min_x) as u32;
        let box_height = (max_y - min_y) as u32;

        if box_width * box_height < min_area {
            continue;
        }

        // Expand bounding box
        let expand_w = (box_width as f32 * expand_ratio * 0.5) as i32;
        let expand_h = (box_height as f32 * expand_ratio * 0.5) as i32;

        let final_x = (min_x - expand_w).max(0) as u32;
        let final_y = (min_y - expand_h).max(0) as u32;
        let final_w = ((max_x + expand_w) as u32)
            .min(width)
            .saturating_sub(final_x);
        let final_h = ((max_y + expand_h) as u32)
            .min(height)
            .saturating_sub(final_y);

        if final_w > 0 && final_h > 0 {
            let rect = Rect::at(final_x as i32, final_y as i32).of_size(final_w, final_h);
            boxes.push(TextBox::new(rect, 1.0));
        }
    }

    // 5. Merge adjacent boxes to form text lines
    merge_into_text_lines(&boxes, 10)
}

/// OTSU adaptive threshold calculation
fn otsu_threshold(image: &GrayImage) -> u8 {
    // Calculate histogram
    let mut histogram = [0u32; 256];
    for pixel in image.pixels() {
        histogram[pixel.0[0] as usize] += 1;
    }

    let total = image.pixels().count() as f64;
    let mut sum = 0.0;
    for (i, &count) in histogram.iter().enumerate() {
        sum += i as f64 * count as f64;
    }

    let mut sum_b = 0.0;
    let mut w_b = 0.0;
    let mut max_variance = 0.0;
    let mut threshold = 0u8;

    for (t, &count) in histogram.iter().enumerate() {
        w_b += count as f64;
        if w_b == 0.0 {
            continue;
        }

        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }

        sum_b += t as f64 * count as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;

        let variance = w_b * w_f * (m_b - m_f).powi(2);
        if variance > max_variance {
            max_variance = variance;
            threshold = t as u8;
        }
    }

    threshold
}

/// Merge independent character boxes into text lines
fn merge_into_text_lines(boxes: &[TextBox], gap_threshold: i32) -> Vec<TextBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // Group by y coordinate
    let mut sorted_boxes: Vec<_> = boxes.iter().collect();
    sorted_boxes.sort_by_key(|b| b.rect.top());

    let mut lines: Vec<TextBox> = Vec::new();

    for bbox in sorted_boxes {
        let mut merged = false;

        // Try to merge into existing lines
        for line in &mut lines {
            let line_center_y = line.rect.top() + line.rect.height() as i32 / 2;
            let box_center_y = bbox.rect.top() + bbox.rect.height() as i32 / 2;

            // If vertical overlap and horizontal proximity
            if (line_center_y - box_center_y).abs() < line.rect.height() as i32 / 2 {
                let line_right = line.rect.left() + line.rect.width() as i32;
                let box_left = bbox.rect.left();

                if (box_left - line_right).abs() < gap_threshold * 3 {
                    // Merge
                    let new_left = line.rect.left().min(bbox.rect.left());
                    let new_top = line.rect.top().min(bbox.rect.top());
                    let new_right = (line.rect.left() + line.rect.width() as i32)
                        .max(bbox.rect.left() + bbox.rect.width() as i32);
                    let new_bottom = (line.rect.top() + line.rect.height() as i32)
                        .max(bbox.rect.top() + bbox.rect.height() as i32);

                    line.rect = Rect::at(new_left, new_top)
                        .of_size((new_right - new_left) as u32, (new_bottom - new_top) as u32);
                    merged = true;
                    break;
                }
            }
        }

        if !merged {
            lines.push(bbox.clone());
        }
    }

    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_textbox_new() {
        let rect = Rect::at(10, 20).of_size(100, 50);
        let tb = TextBox::new(rect, 0.95);

        assert_eq!(tb.rect.left(), 10);
        assert_eq!(tb.rect.top(), 20);
        assert_eq!(tb.rect.width(), 100);
        assert_eq!(tb.rect.height(), 50);
        assert_eq!(tb.score, 0.95);
        assert!(tb.points.is_none());
    }

    #[test]
    fn test_textbox_with_points() {
        let rect = Rect::at(0, 0).of_size(100, 50);
        let points = [
            Point::new(0.0, 0.0),
            Point::new(100.0, 0.0),
            Point::new(100.0, 50.0),
            Point::new(0.0, 50.0),
        ];
        let tb = TextBox::with_points(rect, 0.9, points);

        assert!(tb.points.is_some());
        let pts = tb.points.unwrap();
        assert_eq!(pts[0].x, 0.0);
        assert_eq!(pts[1].x, 100.0);
    }

    #[test]
    fn test_textbox_area() {
        let tb = TextBox::new(Rect::at(0, 0).of_size(100, 50), 0.9);
        assert_eq!(tb.area(), 5000);
    }

    #[test]
    fn test_textbox_expand() {
        let tb = TextBox::new(Rect::at(50, 50).of_size(100, 100), 0.9);
        let expanded = tb.expand(10, 500, 500);

        assert_eq!(expanded.rect.left(), 40);
        assert_eq!(expanded.rect.top(), 40);
        assert_eq!(expanded.rect.width(), 120);
        assert_eq!(expanded.rect.height(), 120);
    }

    #[test]
    fn test_textbox_expand_clamp() {
        // 测试边界裁剪
        let tb = TextBox::new(Rect::at(5, 5).of_size(100, 100), 0.9);
        let expanded = tb.expand(10, 200, 200);

        // 左上角应该被限制在 (0, 0)
        assert_eq!(expanded.rect.left(), 0);
        assert_eq!(expanded.rect.top(), 0);
    }

    #[test]
    fn test_compute_iou() {
        let a = Rect::at(0, 0).of_size(10, 10);
        let b = Rect::at(5, 5).of_size(10, 10);

        let iou = compute_iou(&a, &b);
        assert!(iou > 0.0 && iou < 1.0);

        // 不相交
        let c = Rect::at(100, 100).of_size(10, 10);
        assert_eq!(compute_iou(&a, &c), 0.0);

        // 完全重叠
        assert_eq!(compute_iou(&a, &a), 1.0);
    }

    #[test]
    fn test_compute_iou_partial_overlap() {
        // 50% 重叠的情况
        let a = Rect::at(0, 0).of_size(10, 10);
        let b = Rect::at(5, 0).of_size(10, 10);

        let iou = compute_iou(&a, &b);
        // 交集面积 = 5 * 10 = 50
        // 并集面积 = 100 + 100 - 50 = 150
        // IoU = 50 / 150 ≈ 0.333
        assert!((iou - 0.333).abs() < 0.01);
    }

    #[test]
    fn test_nms() {
        // 第一个和第二个框有很大重叠，第三个框独立
        let boxes = vec![
            TextBox::new(Rect::at(0, 0).of_size(10, 10), 0.9),
            TextBox::new(Rect::at(1, 1).of_size(10, 10), 0.8), // 与第一个框高度重叠
            TextBox::new(Rect::at(100, 100).of_size(10, 10), 0.7),
        ];

        let result = nms(&boxes, 0.3); // 使用较低的阈值确保重叠框被过滤
                                       // 第一个框（最高分数）和第三个框（无重叠）应该保留
        assert!(
            result.len() >= 2,
            "至少应该保留2个框，实际: {}",
            result.len()
        );
    }

    #[test]
    fn test_nms_empty() {
        let boxes: Vec<TextBox> = vec![];
        let result = nms(&boxes, 0.5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_nms_single() {
        let boxes = vec![TextBox::new(Rect::at(0, 0).of_size(10, 10), 0.9)];
        let result = nms(&boxes, 0.5);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_nms_no_overlap() {
        let boxes = vec![
            TextBox::new(Rect::at(0, 0).of_size(10, 10), 0.9),
            TextBox::new(Rect::at(50, 50).of_size(10, 10), 0.8),
            TextBox::new(Rect::at(100, 100).of_size(10, 10), 0.7),
        ];

        let result = nms(&boxes, 0.5);
        assert_eq!(result.len(), 3); // 所有框都保留
    }

    #[test]
    fn test_merge_adjacent() {
        let boxes = vec![
            TextBox::new(Rect::at(0, 0).of_size(10, 10), 1.0),
            TextBox::new(Rect::at(12, 0).of_size(10, 10), 1.0), // 水平距离 2
            TextBox::new(Rect::at(100, 100).of_size(10, 10), 1.0),
        ];

        let result = merge_adjacent_boxes(&boxes, 5);
        assert_eq!(result.len(), 2); // 前两个应该合并
    }

    #[test]
    fn test_merge_adjacent_empty() {
        let boxes: Vec<TextBox> = vec![];
        let result = merge_adjacent_boxes(&boxes, 5);
        assert!(result.is_empty());
    }

    #[test]
    fn test_sort_boxes_by_reading_order() {
        let mut boxes = vec![
            TextBox::new(Rect::at(100, 0).of_size(10, 10), 0.9), // 第一行右边
            TextBox::new(Rect::at(0, 0).of_size(10, 10), 0.9),   // 第一行左边
            TextBox::new(Rect::at(0, 50).of_size(10, 10), 0.9),  // 第二行
        ];

        sort_boxes_by_reading_order(&mut boxes);

        // 应该先按行排序，然后行内按x坐标排序
        assert_eq!(boxes[0].rect.left(), 0);
        assert_eq!(boxes[0].rect.top(), 0);
    }

    #[test]
    fn test_group_boxes_by_line() {
        let boxes = vec![
            TextBox::new(Rect::at(0, 0).of_size(50, 20), 0.9),
            TextBox::new(Rect::at(60, 0).of_size(50, 20), 0.9),
            TextBox::new(Rect::at(0, 50).of_size(50, 20), 0.9),
        ];

        let lines = group_boxes_by_line(&boxes, 10);

        // 应该分成两行
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn test_extract_boxes_without_unclip_is_tighter_than_unclip() {
        let width = 10;
        let height = 10;
        let mut mask = vec![0u8; (width * height) as usize];

        for y in 2..6 {
            for x in 3..7 {
                mask[(y * width + x) as usize] = 255;
            }
        }

        let raw_boxes = extract_boxes_from_mask_without_unclip(
            &mask, width, height, width, height, width, height, 1,
        );
        let unclipped_boxes =
            extract_boxes_with_unclip(&mask, width, height, width, height, width, height, 1, 1.5);

        assert_eq!(raw_boxes.len(), 1);
        assert_eq!(unclipped_boxes.len(), 1);

        let raw = &raw_boxes[0].rect;
        let expanded = &unclipped_boxes[0].rect;

        assert!(expanded.width() >= raw.width());
        assert!(expanded.height() >= raw.height());
        assert!(expanded.left() <= raw.left());
        assert!(expanded.top() <= raw.top());
    }
}
