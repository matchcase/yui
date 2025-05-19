import random
from typing import Optional
from tools import register_tool
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class RollDiceArgs(BaseModel):
    number_of_dice: Optional[int] = Field(description="Number of die to be rolled.",
                                          default=1)
    number_of_faces: Optional[int] = Field(description="Number of faces on each dice.",
                                           default=6)

@register_tool(
    name="roll_dice",
    description="Rolls dice based on arguments.",
    args_schema=RollDiceArgs # Schema for LLM to fill
)

def roll_dice(number_of_dice: int = 1, number_of_faces: int = 6) -> str:
    """Rolls dice, with optional arguments for the number of die and the number of faces on each dice."""
    logger.info(f"Executing roll_dice with {number_of_dice} die with {number_of_faces} faces each.")
    
    rolls = [random.randint(1, number_of_faces) for _ in range(number_of_dice)]
    total = sum(rolls)
    
    result = f"Total Result: {total}\nIndividual Values: {rolls}"
    
    logger.info(f"Dice roll result: {total}, details: {rolls}")
    return result
