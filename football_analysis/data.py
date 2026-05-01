from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "FootballAnalysisML" / "players_18.csv"
LEGACY_CLASSIFICATION_PATH = (
    PROJECT_ROOT / "FootballAnalysisML" / "players_18_processed_classification.csv"
)

NUMERIC_FEATURES = [
    "age",
    "overall",
    "height_cm",
    "weight_kg",
    "potential",
    "release_clause_eur",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",
    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",
    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",
    "defending_marking_awareness",
    "defending_standing_tackle",
    "defending_sliding_tackle",
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
    "goalkeeping_speed",
]

CATEGORICAL_FEATURES = [
    "preferred_foot",
    "work_rate",
    "weak_foot",
    "skill_moves",
    "international_reputation",
]

ROLE_ATTACK = {"LS", "ST", "RS", "LW", "LF", "CF", "RF", "RW"}
ROLE_MIDFIELD = {
    "LAM",
    "CAM",
    "RAM",
    "LM",
    "LCM",
    "CM",
    "RCM",
    "RM",
    "LWB",
    "LDM",
    "CDM",
    "RDM",
}
ROLE_DEFENSE = {"RWB", "LB", "LCB", "CB", "RCB", "RB"}
ROLE_GOALKEEPER = {"GK"}

POSITION_MERGES = {"CF": "ST", "RWB": "RB", "LWB": "LB"}


def load_raw_data(path: Path | str | None = None) -> pd.DataFrame:
    data_path = Path(path) if path is not None else DEFAULT_DATA_PATH
    return pd.read_csv(data_path, low_memory=False)


def extract_primary_position(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.split(",")[0].strip()


def map_role_target(primary_position: str | None) -> str | None:
    if primary_position in ROLE_ATTACK:
        return "attack"
    if primary_position in ROLE_MIDFIELD:
        return "midfield"
    if primary_position in ROLE_DEFENSE:
        return "defense"
    if primary_position in ROLE_GOALKEEPER:
        return "goalkeeper"
    return None


def map_position_target(primary_position: str | None) -> str | None:
    if primary_position is None:
        return None
    return POSITION_MERGES.get(primary_position, primary_position)


def add_targets(data: pd.DataFrame) -> pd.DataFrame:
    frame = data.copy()
    frame["primary_position"] = frame["player_positions"].map(extract_primary_position)
    frame["role_target"] = frame["primary_position"].map(map_role_target)
    frame["position_target"] = frame["primary_position"].map(map_position_target)
    return frame


def _missing_indicator_columns(data: pd.DataFrame) -> list[str]:
    return [col for col in NUMERIC_FEATURES if data[col].isna().any()]


def build_model_frame(
    data: pd.DataFrame,
    task: str,
) -> tuple[pd.DataFrame, list[str], list[str], str]:
    if task not in {"role", "position"}:
        raise ValueError(f"Unsupported task: {task}")

    with_targets = add_targets(data)
    target_col = "role_target" if task == "role" else "position_target"
    frame = with_targets[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [target_col]].copy()
    frame = frame.dropna(subset=[target_col]).reset_index(drop=True)

    indicator_cols: list[str] = []
    for col in _missing_indicator_columns(frame):
        indicator_col = f"{col}_missing"
        frame[indicator_col] = frame[col].isna().astype(int)
        indicator_cols.append(indicator_col)

    numeric_features = list(NUMERIC_FEATURES) + indicator_cols
    categorical_features = list(CATEGORICAL_FEATURES)
    return frame, numeric_features, categorical_features, target_col


def stratified_sample(
    frame: pd.DataFrame,
    target_col: str,
    sample_size: int,
    random_state: int,
) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(frame):
        return frame

    sampled_parts = []
    grouped = frame.groupby(target_col, sort=False)
    for _, part in grouped:
        target_rows = max(1, round(len(part) * sample_size / len(frame)))
        sampled_parts.append(
            part.sample(n=min(target_rows, len(part)), random_state=random_state)
        )
    sampled = pd.concat(sampled_parts).sort_index()
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_state)
    elif len(sampled) < sample_size:
        remaining = frame.drop(index=sampled.index)
        extra = remaining.sample(
            n=min(sample_size - len(sampled), len(remaining)),
            random_state=random_state,
        )
        sampled = pd.concat([sampled, extra]).sort_index()
    return sampled.reset_index(drop=True)


def iter_position_examples(values: Iterable[object], limit: int = 5) -> list[str]:
    examples: list[str] = []
    for value in values:
        primary = extract_primary_position(value)
        if primary is None:
            continue
        examples.append(primary)
        if len(examples) >= limit:
            break
    return examples
