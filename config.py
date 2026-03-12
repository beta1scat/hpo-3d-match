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


# ---------------------------------------------------------------------------
# Search space definition (Table 2 in paper)
# ---------------------------------------------------------------------------
SEARCH_SPACE = {
    "RelSamplingDistance": {"type": "float", "low": 0.03, "high": 0.1, "step": 0.01},
    "KeyPointFraction": {"type": "float", "low": 0.05, "high": 0.3, "step": 0.01},
    "max_overlap_dist_rel": {"type": "float", "low": 0.1, "high": 1.0, "step": 0.1},
    "pose_ref_num_steps": {"type": "int", "low": 1, "high": 20, "step": 1},
    "pose_ref_sub_sampling": {"type": "int", "low": 1, "high": 10, "step": 1},
    "pose_ref_dist_threshold_rel": {
        "type": "float",
        "low": 0.03,
        "high": 0.2,
        "step": 0.01,
    },
    "pose_ref_scoring_dist_rel": {
        "type": "categorical",
        "choices": [0.2, 0.01, 0.005, 0.0001],
    },
    "pose_ref_use_scene_normals": {"type": "categorical", "choices": ["true", "false"]},
}

# Default parameter values from HALCON documentation
DEFAULT_PARAMS = {
    "RelSamplingDistance": 0.05,
    "KeyPointFraction": 0.2,
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


# ---------------------------------------------------------------------------
# Sampler and pruner combinations (Section 3 / Algorithm 1)
# ---------------------------------------------------------------------------
SAMPLER_NAMES = ["TPE", "CmaEs", "NSGAII", "QMC", "Random"]
PRUNER_NAMES = ["Median", "Nop", "Hyperband"]


MODEL_FILE_NAMES = {
    "star": "star.ply",
    "screw_black": "screw.ply",
    "bracket_planar": "bracket_planar.ply",
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
