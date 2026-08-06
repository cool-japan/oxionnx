//! NonMaxSuppression: IoU-based box suppression for object detection

use oxionnx_core::Tensor;

/// Compute IoU between two boxes
fn iou(box_a: [f32; 4], box_b: [f32; 4], center_point_box: i64) -> f32 {
    let (y1_a, x1_a, y2_a, x2_a) = if center_point_box == 1 {
        // center_x, center_y, width, height format
        (
            box_a[1] - box_a[3] / 2.0,
            box_a[0] - box_a[2] / 2.0,
            box_a[1] + box_a[3] / 2.0,
            box_a[0] + box_a[2] / 2.0,
        )
    } else {
        // y1, x1, y2, x2 format
        (
            box_a[0].min(box_a[2]),
            box_a[1].min(box_a[3]),
            box_a[0].max(box_a[2]),
            box_a[1].max(box_a[3]),
        )
    };
    let (y1_b, x1_b, y2_b, x2_b) = if center_point_box == 1 {
        (
            box_b[1] - box_b[3] / 2.0,
            box_b[0] - box_b[2] / 2.0,
            box_b[1] + box_b[3] / 2.0,
            box_b[0] + box_b[2] / 2.0,
        )
    } else {
        (
            box_b[0].min(box_b[2]),
            box_b[1].min(box_b[3]),
            box_b[0].max(box_b[2]),
            box_b[1].max(box_b[3]),
        )
    };

    let inter_y1 = y1_a.max(y1_b);
    let inter_x1 = x1_a.max(x1_b);
    let inter_y2 = y2_a.min(y2_b);
    let inter_x2 = x2_a.min(x2_b);

    let inter_area = (inter_y2 - inter_y1).max(0.0) * (inter_x2 - inter_x1).max(0.0);
    let area_a = (y2_a - y1_a) * (x2_a - x1_a);
    let area_b = (y2_b - y1_b) * (x2_b - x1_b);
    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

/// NonMaxSuppression
/// boxes: [num_batches, num_boxes, 4]
/// scores: [num_batches, num_classes, num_boxes]
/// Returns: [num_selected, 3] with columns (batch_index, class_index, box_index)
pub fn non_max_suppression(
    boxes: &Tensor,
    scores: &Tensor,
    max_output_per_class: usize,
    iou_threshold: f32,
    score_threshold: f32,
    center_point_box: i64,
) -> Result<Tensor, String> {
    if boxes.ndim() != 3 || boxes.shape[2] != 4 {
        return Err("nms: boxes must be [batches, num_boxes, 4]".into());
    }
    if scores.ndim() != 3 {
        return Err("nms: scores must be [batches, num_classes, num_boxes]".into());
    }

    let num_batches = boxes.shape[0];
    let num_boxes = boxes.shape[1];
    let num_classes = scores.shape[1];

    // ONNX spec: max_output_boxes_per_class defaults to 0, and 0 means "select
    // no boxes" for that class -- it is NOT a sentinel for "unlimited". The
    // caller (registry/misc_ops.rs) already passes 0 when the optional input
    // is absent, matching the spec default, so we must honor it literally.
    let max_out = max_output_per_class;

    let mut selected: Vec<[f32; 3]> = Vec::new();

    for b in 0..num_batches {
        for c in 0..num_classes {
            // Get scores for this batch+class
            let mut scored_indices: Vec<(usize, f32)> = (0..num_boxes)
                .map(|i| (i, scores.data[(b * num_classes + c) * num_boxes + i]))
                .filter(|&(_, s)| s > score_threshold)
                .collect();

            // Sort by score descending
            scored_indices.sort_by(|a, b_item| {
                b_item
                    .1
                    .partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut keep: Vec<usize> = Vec::new();
            let mut suppressed = vec![false; num_boxes];

            for &(idx, _) in &scored_indices {
                if suppressed[idx] {
                    continue;
                }
                if keep.len() >= max_out {
                    break;
                }

                keep.push(idx);

                // Suppress overlapping boxes
                let box_a = [
                    boxes.data[(b * num_boxes + idx) * 4],
                    boxes.data[(b * num_boxes + idx) * 4 + 1],
                    boxes.data[(b * num_boxes + idx) * 4 + 2],
                    boxes.data[(b * num_boxes + idx) * 4 + 3],
                ];

                for &(other_idx, _) in &scored_indices {
                    if suppressed[other_idx] || other_idx == idx {
                        continue;
                    }
                    let box_b = [
                        boxes.data[(b * num_boxes + other_idx) * 4],
                        boxes.data[(b * num_boxes + other_idx) * 4 + 1],
                        boxes.data[(b * num_boxes + other_idx) * 4 + 2],
                        boxes.data[(b * num_boxes + other_idx) * 4 + 3],
                    ];
                    if iou(box_a, box_b, center_point_box) > iou_threshold {
                        suppressed[other_idx] = true;
                    }
                }
            }

            for &idx in &keep {
                selected.push([b as f32, c as f32, idx as f32]);
            }
        }
    }

    let num_selected = selected.len();
    if num_selected == 0 {
        return Ok(Tensor::new(Vec::new(), vec![0, 3]));
    }
    let data: Vec<f32> = selected.into_iter().flat_map(|s| s.into_iter()).collect();
    Ok(Tensor::new(data, vec![num_selected, 3]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nms_basic() {
        // 1 batch, 1 class, 3 boxes (y1,x1,y2,x2 format)
        let boxes = Tensor::new(
            vec![
                0.0, 0.0, 1.0, 1.0, // box 0
                0.1, 0.1, 1.1, 1.1, // box 1 (overlaps with box 0)
                2.0, 2.0, 3.0, 3.0, // box 2 (no overlap)
            ],
            vec![1, 3, 4],
        );
        let scores = Tensor::new(vec![0.9, 0.8, 0.7], vec![1, 1, 3]);
        let out = non_max_suppression(&boxes, &scores, 10, 0.5, 0.0, 0).expect("nms basic failed");
        // Box 0 should be selected (highest score)
        // Box 1 should be suppressed (IoU > 0.5 with box 0)
        // Box 2 should be selected (no overlap)
        assert_eq!(out.shape[0], 2); // 2 boxes selected
        assert_eq!(out.shape[1], 3);
    }

    #[test]
    fn test_nms_score_threshold() {
        let boxes = Tensor::new(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], vec![1, 2, 4]);
        let scores = Tensor::new(vec![0.9, 0.1], vec![1, 1, 2]);
        let out = non_max_suppression(&boxes, &scores, 10, 0.5, 0.5, 0)
            .expect("nms score threshold failed");
        assert_eq!(out.shape[0], 1); // Only box with score > 0.5
    }

    #[test]
    fn test_nms_max_output() {
        let boxes = Tensor::new(
            vec![
                0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0, 10.0, 10.0, 11.0, 11.0,
            ],
            vec![1, 3, 4],
        );
        let scores = Tensor::new(vec![0.9, 0.8, 0.7], vec![1, 1, 3]);
        let out =
            non_max_suppression(&boxes, &scores, 1, 0.5, 0.0, 0).expect("nms max output failed");
        assert_eq!(out.shape[0], 1); // max 1 per class
    }

    #[test]
    fn test_nms_max_output_per_class_zero_selects_nothing() {
        // ONNX spec: max_output_boxes_per_class default (and explicit) value
        // of 0 means "select zero boxes for this class", not "unlimited".
        // registry/misc_ops.rs passes 0 when the optional input is absent,
        // so this exercises the exact default-omitted-input scenario.
        let boxes = Tensor::new(
            vec![
                0.0, 0.0, 1.0, 1.0, 5.0, 5.0, 6.0, 6.0, 10.0, 10.0, 11.0, 11.0,
            ],
            vec![1, 3, 4],
        );
        let scores = Tensor::new(vec![0.9, 0.8, 0.7], vec![1, 1, 3]);
        let out =
            non_max_suppression(&boxes, &scores, 0, 0.5, 0.0, 0).expect("nms max_output=0 failed");
        assert_eq!(
            out.shape,
            vec![0, 3],
            "max_output_per_class=0 must select zero boxes"
        );
        assert!(out.data.is_empty());
    }

    #[test]
    fn test_nms_no_boxes_above_threshold() {
        let boxes = Tensor::new(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], vec![1, 2, 4]);
        let scores = Tensor::new(vec![0.1, 0.2], vec![1, 1, 2]);
        let out =
            non_max_suppression(&boxes, &scores, 10, 0.5, 0.9, 0).expect("nms no boxes failed");
        assert_eq!(out.shape[0], 0);
    }

    #[test]
    fn test_nms_center_format() {
        // center_x, center_y, width, height format
        let boxes = Tensor::new(
            vec![
                0.5, 0.5, 1.0, 1.0, // box 0: center (0.5,0.5), w=1, h=1
                0.6, 0.6, 1.0, 1.0, // box 1: overlaps with box 0
                5.0, 5.0, 1.0, 1.0, // box 2: far away
            ],
            vec![1, 3, 4],
        );
        let scores = Tensor::new(vec![0.9, 0.8, 0.7], vec![1, 1, 3]);
        let out = non_max_suppression(&boxes, &scores, 10, 0.5, 0.0, 1)
            .expect("nms center format failed");
        assert_eq!(out.shape[0], 2);
    }

    #[test]
    fn test_nms_invalid_boxes_shape() {
        let boxes = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 1, 3]);
        let scores = Tensor::new(vec![0.9], vec![1, 1, 1]);
        let result = non_max_suppression(&boxes, &scores, 10, 0.5, 0.0, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_iou_no_overlap() {
        let val = iou([0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0], 0);
        assert!((val - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_iou_full_overlap() {
        let val = iou([0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0], 0);
        assert!((val - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_iou_partial_overlap() {
        let val = iou([0.0, 0.0, 2.0, 2.0], [1.0, 1.0, 3.0, 3.0], 0);
        // intersection: 1x1=1, union: 4+4-1=7
        assert!((val - 1.0 / 7.0).abs() < 1e-5);
    }
}
