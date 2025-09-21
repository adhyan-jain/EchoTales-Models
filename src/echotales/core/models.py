from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class Gender(str, Enum):
    MALE = "male"
    FEMALE = "female"
    UNKNOWN = "unknown"


class AgeGroup(str, Enum):
    CHILD = "child"
    TEENAGER = "teenager"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    ELDERLY = "elderly"


class PersonalityVector(BaseModel):
    openness: float = Field(ge=0.0, le=1.0)
    conscientiousness: float = Field(ge=0.0, le=1.0)
    extraversion: float = Field(ge=0.0, le=1.0)
    agreeableness: float = Field(ge=0.0, le=1.0)
    neuroticism: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class PhysicalDescription(BaseModel):
    height: Optional[str] = None
    build: Optional[str] = None
    hair_color: Optional[str] = None
    hair_style: Optional[str] = None
    eye_color: Optional[str] = None
    skin_tone: Optional[str] = None
    distinctive_features: List[str] = Field(default_factory=list)
    clothing_style: Optional[str] = None
    age_appearance: Optional[str] = None


class VoiceProfile(BaseModel):
    actor_id: str
    gender: Gender
    age_group: AgeGroup
    accent: Optional[str] = None
    pitch: float = Field(ge=0.0, le=1.0)
    tempo: float = Field(ge=0.0, le=1.0)
    emotion: str = "neutral"
    quality_score: float = Field(ge=0.0, le=1.0)
    sample_paths: List[str] = Field(default_factory=list)


class BackgroundSetting(BaseModel):
    setting_id: str
    name: str
    type: str  # "indoor", "outdoor", "fantasy", "modern", etc.
    description: str
    atmosphere: str  # "dark", "bright", "mysterious", etc.
    time_of_day: Optional[str] = None
    weather: Optional[str] = None
    realm: str = "real_world"  # "real_world", "fantasy_realm", "dream", etc.
    image_prompt: str
    

class SettingTransition(BaseModel):
    from_setting: str
    to_setting: str
    chapter: int
    paragraph: int
    transition_text: str
    transition_type: str  # "travel", "teleport", "scene_change", etc.


class Character(BaseModel):
    character_id: str
    name: str
    booknlp_id: int
    gender: Gender
    age_group: AgeGroup
    personality: PersonalityVector
    physical_description: PhysicalDescription
    voice_profile: Optional[VoiceProfile] = None
    importance_score: float = Field(ge=0.0, le=1.0)
    first_appearance: int  # chapter number
    last_appearance: int   # chapter number
    total_dialogue_lines: int = 0
    character_arc: List[str] = Field(default_factory=list)
    relationships: Dict[str, str] = Field(default_factory=dict)
    portrait_url: Optional[str] = None


class DialogueLine(BaseModel):
    line_id: str
    character_id: str
    text: str
    emotion: str
    chapter: int
    paragraph: int
    setting_id: Optional[str] = None
    audio_file: Optional[str] = None
    duration: Optional[float] = None


class Scene(BaseModel):
    scene_id: str
    chapter: int
    start_paragraph: int
    end_paragraph: int
    setting: BackgroundSetting
    characters_present: List[str]
    dialogue_lines: List[DialogueLine]
    narrative_summary: str
    mood: str
    image_url: Optional[str] = None


class ProcessedNovel(BaseModel):
    book_id: str
    title: str
    author: Optional[str] = None
    characters: List[Character]
    scenes: List[Scene]
    settings: List[BackgroundSetting]
    setting_transitions: List[SettingTransition]
    total_chapters: int
    processing_timestamp: datetime
    audio_files: Dict[int, str] = Field(default_factory=dict)  # chapter -> file_path
    metadata: Dict[str, Any] = Field(default_factory=dict)