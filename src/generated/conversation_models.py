from typing import Optional, Literal
from pydantic import BaseModel, Field
import typing

MODELS = {}

class RoutingModel(BaseModel):
    conversation_category: Optional[Literal['task_related', 'greeting', 'help_request', 'off_topic', 'exit', 'unknown']] = Field(default="unknown", description="The general intent or category of the user's message")
    affirmation: Optional[Literal['confirmed', 'declined', 'unknown']] = Field(default="unknown", description="Whether the user is confirming or declining")
MODELS['routing'] = RoutingModel

class TaskRelatedModel(BaseModel):
    intent: Optional[Literal['configure-lighting', 'configure-climate', 'configure-security', 'unknown']] = Field(default="unknown", description="The specific smart home configuration task")
    room: Optional[Literal['living-room', 'bedroom', 'kitchen', 'bathroom', 'unknown']] = Field(default="unknown", description="The room the lighting is for")
    budget: Optional[Literal['low', 'medium', 'high', 'unknown']] = Field(default="unknown", description="Budget constraints")
    automation_level: Optional[Literal['manual', 'scheduled', 'reactive', 'unknown']] = Field(default="unknown", description="Level of automation desired")
MODELS['task_related'] = TaskRelatedModel

class SecurityRelatedModel(BaseModel):
    type: Optional[Literal['cameras', 'locks', 'full-system', 'unknown']] = Field(default="unknown", description="The type of security setup")
    budget: Optional[Literal['low', 'medium', 'high', 'unknown']] = Field(default="unknown", description="Budget for security")
MODELS['security_related'] = SecurityRelatedModel

class ClimateRelatedModel(BaseModel):
    room: Optional[Literal['living-room', 'bedroom', 'whole-house', 'unknown']] = Field(default="unknown", description="The room the climate control is for")
    budget: Optional[Literal['low', 'medium', 'high', 'unknown']] = Field(default="unknown", description="The user budget constraint")
MODELS['climate_related'] = ClimateRelatedModel