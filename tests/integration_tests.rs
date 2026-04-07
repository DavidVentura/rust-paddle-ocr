//! 集成测试
//!
//! Integration Tests
//!
//! 这些测试需要模型文件才能运行

use ocr_rs::{
    DetModel, DetOptions, DetPrecisionMode, OcrEngine, OcrEngineConfig, RecModel, RecOptions,
};

/// 测试模型文件路径
const DET_MODEL_PATH: &str = "models/PP-OCRv5_mobile_det.mnn";
const REC_MODEL_PATH: &str = "models/PP-OCRv5_mobile_rec.mnn";
const CHARSET_PATH: &str = "models/ppocr_keys_v5.txt";
const TEST_IMAGE_PATH: &str = "res/test1.png";

/// 检查模型文件是否存在
fn models_exist() -> bool {
    std::path::Path::new(DET_MODEL_PATH).exists()
        && std::path::Path::new(REC_MODEL_PATH).exists()
        && std::path::Path::new(CHARSET_PATH).exists()
}

/// 检查测试图像是否存在
fn test_image_exists() -> bool {
    std::path::Path::new(TEST_IMAGE_PATH).exists()
}

#[test]
fn test_det_model_creation() {
    if !models_exist() {
        eprintln!("跳过测试：模型文件不存在");
        return;
    }

    let det = DetModel::from_file(DET_MODEL_PATH, None);
    assert!(det.is_ok(), "检测模型创建失败: {:?}", det.err());
}

#[test]
fn test_det_model_with_options() {
    if !models_exist() {
        eprintln!("跳过测试：模型文件不存在");
        return;
    }

    let det = DetModel::from_file(DET_MODEL_PATH, None).map(|d| {
        d.with_options(
            DetOptions::new()
                .with_max_side_len(1280)
                .with_precision_mode(DetPrecisionMode::Fast),
        )
    });

    assert!(det.is_ok(), "配置检测模型失败: {:?}", det.err());
}

#[test]
fn test_rec_model_creation() {
    if !models_exist() {
        eprintln!("跳过测试：模型文件不存在");
        return;
    }

    let rec = RecModel::from_file(REC_MODEL_PATH, CHARSET_PATH, None);
    assert!(rec.is_ok(), "识别模型创建失败: {:?}", rec.err());
}

#[test]
fn test_rec_model_charset_size() {
    if !models_exist() {
        eprintln!("跳过测试：模型文件不存在");
        return;
    }

    let rec = RecModel::from_file(REC_MODEL_PATH, CHARSET_PATH, None).unwrap();
    let charset_size = rec.charset_size();

    // PP-OCRv5 字符集应该有很多字符
    assert!(
        charset_size > 1000,
        "字符集大小应该大于 1000，实际: {}",
        charset_size
    );
}

#[test]
fn test_rec_model_with_options() {
    if !models_exist() {
        eprintln!("跳过测试：模型文件不存在");
        return;
    }

    let rec = RecModel::from_file(REC_MODEL_PATH, CHARSET_PATH, None)
        .map(|r| r.with_options(RecOptions::new().with_min_score(0.5).with_batch_size(4)));

    assert!(rec.is_ok(), "配置识别模型失败: {:?}", rec.err());
}

#[test]
fn test_ocr_engine_creation() {
    if !models_exist() {
        eprintln!("跳过测试：模型文件不存在");
        return;
    }

    let engine = OcrEngine::new(DET_MODEL_PATH, REC_MODEL_PATH, CHARSET_PATH, None);
    assert!(engine.is_ok(), "OCR 引擎创建失败: {:?}", engine.err());
}

#[test]
fn test_ocr_engine_with_config() {
    if !models_exist() {
        eprintln!("跳过测试：模型文件不存在");
        return;
    }

    let config = OcrEngineConfig::new()
        .with_threads(2)
        .with_det_options(DetOptions::fast())
        .with_rec_options(RecOptions::new().with_min_score(0.3));

    let engine = OcrEngine::new(DET_MODEL_PATH, REC_MODEL_PATH, CHARSET_PATH, Some(config));
    assert!(engine.is_ok(), "配置 OCR 引擎失败: {:?}", engine.err());
}

#[test]
fn test_ocr_engine_presets() {
    if !models_exist() {
        eprintln!("跳过测试：模型文件不存在");
        return;
    }

    // 测试快速模式
    let fast_config = OcrEngineConfig::fast();
    let engine = OcrEngine::new(
        DET_MODEL_PATH,
        REC_MODEL_PATH,
        CHARSET_PATH,
        Some(fast_config),
    );
    assert!(engine.is_ok(), "快速模式创建失败");
}

#[test]
fn test_detection_on_image() {
    if !models_exist() || !test_image_exists() {
        eprintln!("跳过测试：模型或测试图像不存在");
        return;
    }

    let det = DetModel::from_file(DET_MODEL_PATH, None).unwrap();
    let image = image::open(TEST_IMAGE_PATH).unwrap();

    let boxes = det.detect(&image);
    assert!(boxes.is_ok(), "检测失败: {:?}", boxes.err());

    let boxes = boxes.unwrap();
    // 测试图像应该有文本
    assert!(!boxes.is_empty(), "测试图像应该检测到文本");
}

#[test]
fn test_detection_components_on_image() {
    if !models_exist() || !test_image_exists() {
        eprintln!("跳过测试：模型或测试图像不存在");
        return;
    }

    let det = DetModel::from_file(DET_MODEL_PATH, None)
        .unwrap()
        .with_options(
            DetOptions::new()
                .with_unclip_ratio(0.7)
                .with_score_threshold(0.35)
                .with_min_area(8),
        );
    let image = image::open(TEST_IMAGE_PATH).unwrap();

    let boxes = det.detect_components(&image);
    assert!(boxes.is_ok(), "组件检测失败: {:?}", boxes.err());

    let boxes = boxes.unwrap();
    assert!(!boxes.is_empty(), "测试图像应该检测到组件文本框");
}

#[test]
fn test_detection_and_crop() {
    if !models_exist() || !test_image_exists() {
        eprintln!("跳过测试：模型或测试图像不存在");
        return;
    }

    let det = DetModel::from_file(DET_MODEL_PATH, None).unwrap();
    let image = image::open(TEST_IMAGE_PATH).unwrap();

    let detections = det.detect_and_crop(&image);
    assert!(detections.is_ok(), "检测裁剪失败: {:?}", detections.err());

    let detections = detections.unwrap();
    for (cropped, text_box) in &detections {
        // 裁剪的图像应该有效
        assert!(cropped.width() > 0);
        assert!(cropped.height() > 0);
        // 边界框应该有有效的分数
        assert!(text_box.score >= 0.0 && text_box.score <= 1.0);
    }
}

#[test]
fn test_recognition_on_cropped_image() {
    if !models_exist() || !test_image_exists() {
        eprintln!("跳过测试：模型或测试图像不存在");
        return;
    }

    let det = DetModel::from_file(DET_MODEL_PATH, None).unwrap();
    let rec = RecModel::from_file(REC_MODEL_PATH, CHARSET_PATH, None).unwrap();
    let image = image::open(TEST_IMAGE_PATH).unwrap();

    let detections = det.detect_and_crop(&image).unwrap();

    if detections.is_empty() {
        eprintln!("未检测到文本区域，跳过识别测试");
        return;
    }

    let (cropped, _) = &detections[0];
    let result = rec.recognize(cropped);

    assert!(result.is_ok(), "识别失败: {:?}", result.err());

    let result = result.unwrap();
    // 置信度应该在有效范围内
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[test]
fn test_full_ocr_pipeline() {
    if !models_exist() || !test_image_exists() {
        eprintln!("跳过测试：模型或测试图像不存在");
        return;
    }

    let engine = OcrEngine::new(DET_MODEL_PATH, REC_MODEL_PATH, CHARSET_PATH, None).unwrap();
    let image = image::open(TEST_IMAGE_PATH).unwrap();

    let results = engine.recognize(&image);
    assert!(results.is_ok(), "OCR 识别失败: {:?}", results.err());

    let results = results.unwrap();
    // 测试图像应该有文本
    assert!(!results.is_empty(), "测试图像应该识别到文本");

    for result in &results {
        // 每个结果应该有有效的数据
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.bbox.area() > 0);
    }
}

#[test]
fn test_batch_recognition() {
    if !models_exist() || !test_image_exists() {
        eprintln!("跳过测试：模型或测试图像不存在");
        return;
    }

    let det = DetModel::from_file(DET_MODEL_PATH, None).unwrap();
    let rec = RecModel::from_file(REC_MODEL_PATH, CHARSET_PATH, None).unwrap();
    let image = image::open(TEST_IMAGE_PATH).unwrap();

    let detections = det.detect_and_crop(&image).unwrap();

    if detections.len() < 2 {
        eprintln!("检测到的文本区域不足，跳过批量测试");
        return;
    }

    let images: Vec<_> = detections.iter().map(|(img, _)| img.clone()).collect();
    let results = rec.recognize_batch(&images);

    assert!(results.is_ok(), "批量识别失败: {:?}", results.err());

    let results = results.unwrap();
    assert_eq!(results.len(), images.len());
}

#[test]
fn test_det_only_engine() {
    if !models_exist() {
        eprintln!("跳过测试：模型文件不存在");
        return;
    }

    let det_engine = OcrEngine::det_only(DET_MODEL_PATH, None);
    assert!(
        det_engine.is_ok(),
        "仅检测引擎创建失败: {:?}",
        det_engine.err()
    );
}

#[test]
fn test_rec_only_engine() {
    if !models_exist() {
        eprintln!("跳过测试：模型文件不存在");
        return;
    }

    let rec_engine = OcrEngine::rec_only(REC_MODEL_PATH, CHARSET_PATH, None);
    assert!(
        rec_engine.is_ok(),
        "仅识别引擎创建失败: {:?}",
        rec_engine.err()
    );
}
