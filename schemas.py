"""
Database Schemas for Tic-Tac-Toe

Each Pydantic model represents a collection in MongoDB.
Collection name is the lowercase of the class name.

- User -> "user"
- Game -> "game"
"""

from pydantic import BaseModel, Field, EmailStr
from typing import List, Optional, Literal
from datetime import datetime


class User(BaseModel):
    email: EmailStr = Field(..., description="User email (unique)")
    password_hash: str = Field(..., description="BCrypt password hash")
    created_at: Optional[datetime] = Field(default=None)


class Game(BaseModel):
    mode: Literal["public", "private"] = Field(..., description="Game visibility/matchmaking mode")
    ai: bool = Field(False, description="True if playing against AI")
    room_code: Optional[str] = Field(None, description="6-char code for private games")
    players: List[Optional[str]] = Field(default_factory=lambda: [None, None], description="Two user IDs (or None)")
    symbols: List[str] = Field(default_factory=lambda: ["X", "O"], description="Player symbols")
    moves: List[int] = Field(default_factory=list, description="Sequence of cell positions 0..8")
    board: List[Optional[str]] = Field(default_factory=lambda: [None]*9, description="Board state")
    turn: str = Field("X", description="Current turn symbol 'X' or 'O'")
    result: Optional[Literal["X", "O", "draw"]] = Field(None, description="Outcome if finished")
    spectators: List[str] = Field(default_factory=list)
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)
