from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """Schema for the prediction response."""

    class_name: str = Field(
        ..., description="The predicted class name of the skin lesion"
    )
    confidence: float = Field(
        ..., description="The confidence score of the prediction (0-1)"
    )
    probabilities: dict[str, float] = Field(
        ..., description="Dictionary mapping all class names to their probabilities"
    )
