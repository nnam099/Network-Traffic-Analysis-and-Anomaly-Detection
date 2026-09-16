"""Compatibility wrapper for IDS v14. New code lives in the ids package and src/train.py."""

from ids.config import CFG, get_config, resolve_paths, seed_everything
from ids.dataset import (
    KNOWN_ATTACK_CATS, ZERO_DAY_ATTACK_CATS, UNSW_RAW_COLUMNS, SKIP_FILES,
    FlowDS, _find_unsw_csvs, load_unsw_csvs, load_official_unsw_splits,
    normalize_labels, validate_declared_classes, feature_fingerprints,
    prepare_official_splits, assert_training_isolation,
    _encode_categorical_features, _get_numeric_features, engineer_features,
    clean_df, prepare_splits, make_loaders,
)
from ids.models import ResBlock, IDSBackbone, ProjectionHead, AutoEncoder, IDSModel
from ids.losses import FocalLoss, SupConLoss, IDSLoss
from ids.trainer import (
    train_epoch, eval_epoch, _collect_loader_predictions, _format_class_name,
    log_top_confusions, normal_class_index, Trainer, train,
)
from ids.threshold import AdaptiveThreshold, static_threshold
from ids.evaluator import (
    compute_hybrid_meta_score, _hybrid_base_features, fit_hybrid_meta_learner,
    _batch_scores, _batch_gradbp, build_centroids, class_prototype_cosine_similarity,
    calibrate, compute_adaptive_threshold_trace, evaluate_classifier, evaluate_zero_day,
)
from ids.plots import (
    _attack_probs_batch, plot_soc_decision_space, plot_per_class_proper,
    plot_training_curve, plot_threshold_drift, plot_roc_curves, plot_confusion_matrix,
)
from train import save_artifacts, run_full, run_demo, main


if __name__ == '__main__':
    main()
