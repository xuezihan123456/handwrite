"""Age-dependent handwriting characteristic profiles.

Defines handwriting parameters for different developmental stages:
lower elementary, upper elementary, middle school, high school, and adult.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AgeGroup(Enum):
    """Developmental age groups for handwriting simulation."""

    LOWER_ELEMENTARY = "lower_elementary"  # 小学低年级 (6-8)
    UPPER_ELEMENTARY = "upper_elementary"  # 小学高年级 (9-12)
    MIDDLE_SCHOOL = "middle_school"  # 初中 (13-15)
    HIGH_SCHOOL = "high_school"  # 高中 (16-18)
    ADULT = "adult"  # 成人 (18+)


@dataclass(frozen=True)
class AgeProfile:
    """Handwriting parameters for a specific age group.

    Attributes:
        jitter_x: Horizontal position jitter in pixels (random offset).
        jitter_y: Vertical position jitter in pixels (random offset).
        stability: Stroke stability factor (0.0 = very unstable, 1.0 = perfectly stable).
        size_variation: Character size variation ratio (0.0 = uniform, 0.4 = highly varied).
        pressure_variation: Stroke pressure variation (0.0 = uniform, 1.0 = highly varied).
        stroke_connection: Tendency for connected/cursive strokes (0.0 = block, 1.0 = flowing).
        speed_factor: Writing speed factor affecting smoothness (0.5 = slow/careful, 1.5 = fast).
        tilt_variation: Baseline tilt variation in degrees (0.0 = level, 5.0 = very tilted).
        line_straightness: How straight the baseline stays (0.0 = very wavy, 1.0 = ruler-straight).
        description: Human-readable description of this age group.
    """

    jitter_x: float
    jitter_y: float
    stability: float
    size_variation: float
    pressure_variation: float
    stroke_connection: float
    speed_factor: float
    tilt_variation: float
    line_straightness: float
    description: str


# Default profiles for each age group
_PROFILES: dict[AgeGroup, AgeProfile] = {
    AgeGroup.LOWER_ELEMENTARY: AgeProfile(
        jitter_x=4.0,
        jitter_y=5.0,
        stability=0.3,
        size_variation=0.30,
        pressure_variation=0.5,
        stroke_connection=0.0,
        speed_factor=0.5,
        tilt_variation=3.5,
        line_straightness=0.4,
        description="小学低年级：字迹歪扭、大小不一、笔画生硬、基线不稳",
    ),
    AgeGroup.UPPER_ELEMENTARY: AgeProfile(
        jitter_x=2.5,
        jitter_y=3.0,
        stability=0.55,
        size_variation=0.20,
        pressure_variation=0.35,
        stroke_connection=0.1,
        speed_factor=0.7,
        tilt_variation=2.0,
        line_straightness=0.6,
        description="小学高年级：基本工整但仍有歪斜、笔画不够流畅",
    ),
    AgeGroup.MIDDLE_SCHOOL: AgeProfile(
        jitter_x=1.5,
        jitter_y=1.5,
        stability=0.75,
        size_variation=0.12,
        pressure_variation=0.25,
        stroke_connection=0.3,
        speed_factor=0.9,
        tilt_variation=1.0,
        line_straightness=0.8,
        description="初中：书写工整、笔画规范、开始有个人风格",
    ),
    AgeGroup.HIGH_SCHOOL: AgeProfile(
        jitter_x=0.8,
        jitter_y=0.8,
        stability=0.90,
        size_variation=0.08,
        pressure_variation=0.20,
        stroke_connection=0.6,
        speed_factor=1.2,
        tilt_variation=0.5,
        line_straightness=0.9,
        description="高中：连笔增多、书写流畅快速、有明显个人风格",
    ),
    AgeGroup.ADULT: AgeProfile(
        jitter_x=0.5,
        jitter_y=0.5,
        stability=0.95,
        size_variation=0.05,
        pressure_variation=0.15,
        stroke_connection=0.7,
        speed_factor=1.3,
        tilt_variation=0.3,
        line_straightness=0.95,
        description="成人：书写成熟、笔画流畅连贯、风格稳定",
    ),
}


def get_age_profile(age_group: AgeGroup) -> AgeProfile:
    """Return the handwriting profile for the given age group.

    Args:
        age_group: The developmental age group.

    Returns:
        The AgeProfile with handwriting parameters.

    Raises:
        KeyError: If the age group is not defined.
    """
    return _PROFILES[age_group]


def list_age_groups() -> list[AgeGroup]:
    """Return all available age groups in developmental order."""
    return [
        AgeGroup.LOWER_ELEMENTARY,
        AgeGroup.UPPER_ELEMENTARY,
        AgeGroup.MIDDLE_SCHOOL,
        AgeGroup.HIGH_SCHOOL,
        AgeGroup.ADULT,
    ]


def interpolate_profiles(
    profile_a: AgeProfile,
    profile_b: AgeProfile,
    t: float,
) -> AgeProfile:
    """Linearly interpolate between two age profiles.

    Args:
        profile_a: The profile at t=0.
        profile_b: The profile at t=1.
        t: Interpolation factor, clamped to [0, 1].

    Returns:
        A new AgeProfile with interpolated values.
    """
    t = max(0.0, min(1.0, t))

    def _lerp(a: float, b: float) -> float:
        return a + (b - a) * t

    return AgeProfile(
        jitter_x=_lerp(profile_a.jitter_x, profile_b.jitter_x),
        jitter_y=_lerp(profile_a.jitter_y, profile_b.jitter_y),
        stability=_lerp(profile_a.stability, profile_b.stability),
        size_variation=_lerp(profile_a.size_variation, profile_b.size_variation),
        pressure_variation=_lerp(profile_a.pressure_variation, profile_b.pressure_variation),
        stroke_connection=_lerp(profile_a.stroke_connection, profile_b.stroke_connection),
        speed_factor=_lerp(profile_a.speed_factor, profile_b.speed_factor),
        tilt_variation=_lerp(profile_a.tilt_variation, profile_b.tilt_variation),
        line_straightness=_lerp(profile_a.line_straightness, profile_b.line_straightness),
        description=f"插值: {profile_a.description} -> {profile_b.description} (t={t:.2f})",
    )


__all__ = [
    "AgeGroup",
    "AgeProfile",
    "get_age_profile",
    "interpolate_profiles",
    "list_age_groups",
]
