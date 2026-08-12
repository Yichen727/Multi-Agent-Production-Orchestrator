class VisionTags(BaseModel):
    """Structured tags generated from sampled video frames."""
    scene_description: str = Field(
        default="",
        description="One factual sentence describing what is happening in this shot.",
    )
    objects: list[str] = Field(
        default_factory=list,
        description="Salient objects or subjects visibly present in the frames.",
    )

class EventTags(BaseModel):
    """Structured tags for one temporal event."""
    action: str = Field(
        default="",
        description="One factual sentence describing what happens across the frames.",
    )
    state_change: str = Field(
        default="",
        description="What is different between the first and last frame.",
    )

    