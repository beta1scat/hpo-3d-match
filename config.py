"""Configuration for models, search spaces, ROI, and dataset paths.

Model configurations for three ITODD objects:
  - star:            12-fold rotational symmetry (symmetryAngle=30, checkAxis=3)
  - screw_black:     axially symmetric (checkAxis=2)
  - bracket_planar:  no symmetry (checkAxis=3)

Search space follows Table 2 in the paper.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from pathlib import Path


# Supported Optuna Samplers and Pruners (3 Core Samplers: Bayesian, Evolutionary, Random)
SAMPLER_NAMES = ("TPE", "CmaEs", "Random")
PRUNER_NAMES = ("Nop", "Median", "Hyperband")

# ---------------------------------------------------------------------------
# Search space definition (Continuous float + discrete int/categorical, 9 core dimensions)
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    # 1. 空间采样连续控制
    "RelSamplingDistance": {"type": "float", "low": 0.010, "high": 0.20},
    "KeyPointFraction": {"type": "float", "low": 0.010, "high": 0.60},
    "min_score": {"type": "float", "low": 0.0, "high": 0.70},
    "max_overlap_dist_rel": {"type": "float", "low": 0.05, "high": 1.00},

    # 2. ICP 姿态精修与评分控制 (Continuous / Discrete Physical Controls)
    "pose_ref_num_steps": {"type": "int", "low": 1, "high": 50, "step": 1},
    "pose_ref_sub_sampling": {"type": "int", "low": 1, "high": 25, "step": 1},
    "pose_ref_dist_threshold_rel": {"type": "float", "low": 0.005, "high": 0.50},
    "pose_ref_scoring_dist_rel": {"type": "float", "low": 0.0001, "high": 0.25},
    "pose_ref_use_scene_normals": {"type": "categorical", "choices": ["true", "false"]},
}

# Default parameter values strictly matching HALCON official documentation
DEFAULT_PARAMS = {
    "RelSamplingDistance": 0.05,
    "KeyPointFraction": 0.2,
    "min_score": 0.0,
    "max_overlap_dist_rel": 0.5,
    "pose_ref_num_steps": 5,
    "pose_ref_sub_sampling": 2,
    "pose_ref_dist_threshold_rel": 0.1,
    "pose_ref_scoring_dist_rel": 0.005,
    "pose_ref_use_scene_normals": "false",
}


# ---------------------------------------------------------------------------
# ROI configuration (shared across all scenes in ITODD)
# ---------------------------------------------------------------------------
@dataclass
class ROIConfig:
    """Region of Interest box and transform for scene filtering."""

    # ROI pose (translation in meters, rotation in degrees)
    tx: float = 0.009090036154
    ty: float = -0.003862455487
    tz: float = 0.704267919064
    rx: float = 0.0
    ry: float = -7.2081212
    rz: float = -7.8092508
    # Bounding box half-extents (meters)
    x_range: Tuple[float, float] = (-0.22216535 / 2, 0.22216535 / 2)
    y_range: Tuple[float, float] = (-0.22267956 / 2, 0.22267956 / 2)
    z_range: Tuple[float, float] = (-0.078 / 2, 0.078 / 2)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Configuration for a specific object model."""

    name: str
    position_bound: float = 10.0  # mm, position error threshold
    rotation_bound: float = 10.0  # deg, rotation error threshold per axis
    check_axis: int = 3  # number of rotation axes to check
    symmetry_angle: Optional[float] = None  # degrees, for discrete rotational symmetry
    gt_z_offset: float = 0.0  # ground truth Z-axis correction (mm)

    @property
    def penalty_per_miss(self) -> float:
        """Penalty value for each unmatched ground truth pose (Eq. 4 in paper)."""
        return self.position_bound + self.check_axis * self.rotation_bound


# Pre-defined model configurations
MODEL_CONFIGS: Dict[str, ModelConfig] = {
    "star": ModelConfig(
        name="star",
        check_axis=3,
        symmetry_angle=30.0,
        gt_z_offset=-5.68374,  # known GT error in Z-axis
    ),
    "screw_black": ModelConfig(
        name="screw_black",
        check_axis=2,
        symmetry_angle=None,  # fully axially symmetric
    ),
    "bracket_planar": ModelConfig(
        name="bracket_planar",
        check_axis=3,
        symmetry_angle=None,  # no rotational symmetry
        gt_z_offset=-5.68374,  # known GT error in Z-axis (same as star)
    ),
}





MODEL_FILE_NAMES = {
    "star": "star.ply",
    "screw_black": "screw.ply",
    "bracket_planar": "bracket_planar.ply",
}

TARGET_OBJECT_IDS = {
    "screw_black": 24,
    "star": 25,
    "bracket_planar": 5,
}


# ---------------------------------------------------------------------------
# Dataset path helper
# ---------------------------------------------------------------------------
@dataclass
class DatasetPaths:
    """Dataset path configuration.

    Supports both:
      - local repo layout: ``data/cad_models/*.ply`` and ``data/scenes/scene_xxxx/...``
      - original ITODD layout used in the paper
    """

    base_path: str = str(Path(__file__).parent / "data")

    def model_path(self, model_name: str) -> str:
        base = Path(self.base_path)

        local_model = base / "cad_models" / MODEL_FILE_NAMES[model_name]
        if local_model.exists():
            return str(local_model)

        legacy_local_model = base / MODEL_FILE_NAMES[model_name]
        if legacy_local_model.exists():
            return str(legacy_local_model)

        itodd_model = (
            base
            / "base_package"
            / "models"
            / "cad_models"
            / MODEL_FILE_NAMES[model_name]
        )
        if itodd_model.exists():
            return str(itodd_model)

        return str(local_model)

    def scene_image_prefix(self, scene_id: str) -> str:
        base = Path(self.base_path)
        padded_scene = f"scene_{scene_id.zfill(4)}"

        local_prefix = base / "scenes" / padded_scene / "3d_long_baseline"
        if (Path(str(local_prefix) + "_x.tif")).exists():
            return str(local_prefix)

        itodd_prefix = (
            base / "3d_long_baseline" / "scenes" / padded_scene / "3d_long_baseline"
        )
        if (Path(str(itodd_prefix) + "_x.tif")).exists():
            return str(itodd_prefix)

        return str(local_prefix)
