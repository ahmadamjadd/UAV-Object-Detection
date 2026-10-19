"""Metadata describing a registered detection dataset."""

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class DatasetMetadata(BaseModel):
    """Describe dataset identity and layout without reading image files."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        frozen=True,
        validate_default=True,
    )

    dataset_id: str = Field(
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$",
    )

    class_names: tuple[str, ...] = Field(min_length=1)
    train_images_ref: str
    validation_images_ref: str

    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    validation_status: Literal["pending", "valid", "invalid"] = "pending"

    @field_validator("class_names")
    @classmethod
    def check_class_names(cls, names: tuple[str, ...]) -> tuple[str, ...]:
        if any(not name.strip() for name in names):
            raise ValueError("Class names must not be blank")
        if len(names) != len(set(names)):
            raise ValueError("Class names must be unique")
        return names

    @field_validator("train_images_ref", "validation_images_ref")
    @classmethod
    def check_split_reference(cls, value: str) -> str:
        path = PurePosixPath(value)

        if not value.strip():
            raise ValueError("Split references must not be blank")

        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or ":" in value
        ):
            raise ValueError(
                "Use a relative path with forward slashes and no '..'"
            )

        return value