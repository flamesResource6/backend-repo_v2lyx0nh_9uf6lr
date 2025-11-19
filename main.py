import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, Depends, Cookie, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr
from passlib.hash import bcrypt
import jwt
from bson import ObjectId

from database import db

JWT_SECRET = os.getenv("JWT_SECRET", "dev_secret_change_me")
JWT_ALG = "HS256"
COOKIE_NAME = "ttt_token"

app = FastAPI(title="TicTacToe API")

# CORS - allow credentials for cookie-based auth
# Use origin regex so Access-Control-Allow-Origin is the requesting origin (not "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[],
    allow_origin_regex=r".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------
# Utility helpers
# ---------------------------

def oid(obj: Any) -> Optional[ObjectId]:
    try:
        return ObjectId(obj)
    except Exception:
        return None


def serialize_doc(doc: dict) -> dict:
    if not doc:
        return doc
    d = dict(doc)
    if "_id" in d:
        d["id"] = str(d.pop("_id"))
    # Convert any ObjectId in arrays
    if "players" in d:
        d["players"] = [str(p) if isinstance(p, ObjectId) else p for p in d["players"]]
    if "spectators" in d:
        d["spectators"] = [str(s) if isinstance(s, ObjectId) else s for s in d["spectators"]]
    return d


WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # cols
    (0, 4, 8), (2, 4, 6)              # diags
]


def check_winner(board: List[Optional[str]]) -> Optional[str]:
    for a, b, c in WIN_LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    if all(cell is not None for cell in board):
        return "draw"
    return None


def available_moves(board: List[Optional[str]]) -> List[int]:
    return [i for i, v in enumerate(board) if v is None]


def minimax(board: List[Optional[str]], turn: str) -> int:
    winner = check_winner(board)
    if winner == 'X':
        return 1
    if winner == 'O':
        return -1
    if winner == 'draw':
        return 0
    if turn == 'X':
        best = -10
        for m in available_moves(board):
            board[m] = 'X'
            score = minimax(board, 'O')
            board[m] = None
            best = max(best, score)
        return best
    else:
        best = 10
        for m in available_moves(board):
            board[m] = 'O'
            score = minimax(board, 'X')
            board[m] = None
            best = min(best, score)
        return best


def best_ai_move(board: List[Optional[str]], ai_symbol: str) -> int:
    # If AI is 'O', we minimize; if 'X' we maximize
    moves = available_moves(board)
    best_move = moves[0]
    if ai_symbol == 'X':
        best_score = -10
        for m in moves:
            board[m] = 'X'
            score = minimax(board, 'O')
            board[m] = None
            if score > best_score:
                best_score = score
                best_move = m
    else:
        best_score = 10
        for m in moves:
            board[m] = 'O'
            score = minimax(board, 'X')
            board[m] = None
            if score < best_score:
                best_score = score
                best_move = m
    return best_move


# ---------------------------
# Auth
# ---------------------------
class RegisterBody(BaseModel):
    email: EmailStr
    password: str


class LoginBody(BaseModel):
    email: EmailStr
    password: str


def create_token(user_id: str) -> str:
    payload = {
        "sub": user_id,
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)


def set_auth_cookie(resp: JSONResponse, token: str):
    resp.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        secure=True,           # preview runs over https
        samesite="none",       # allow cross-site cookie from preview frontends
        max_age=7*24*3600,
        path="/",
    )


def get_current_user(token: Optional[str] = Cookie(default=None, alias=COOKIE_NAME)) -> Optional[dict]:
    if not token:
        return None
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        uid = payload.get("sub")
        user = db["user"].find_one({"_id": ObjectId(uid)})
        return user
    except Exception:
        return None


@app.post('/api/auth/register')
def register(body: RegisterBody):
    existing = db["user"].find_one({"email": body.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    pw_hash = bcrypt.hash(body.password)
    now = datetime.now(timezone.utc)
    res = db["user"].insert_one({
        "email": body.email,
        "password_hash": pw_hash,
        "created_at": now,
    })
    token = create_token(str(res.inserted_id))
    resp = JSONResponse({"id": str(res.inserted_id), "email": body.email})
    set_auth_cookie(resp, token)
    return resp


@app.post('/api/auth/login')
def login(body: LoginBody):
    user = db["user"].find_one({"email": body.email})
    if not user or not bcrypt.verify(body.password, user.get("password_hash", "")):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token(str(user["_id"]))
    resp = JSONResponse({"id": str(user["_id"]), "email": user["email"]})
    set_auth_cookie(resp, token)
    return resp


@app.post('/api/auth/logout')
def logout():
    resp = JSONResponse({"message": "logged out"})
    resp.delete_cookie(COOKIE_NAME, path="/")
    return resp


@app.get('/api/auth/me')
def me(user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return {"id": str(user["_id"]), "email": user["email"]}


# ---------------------------
# Games
# ---------------------------
class CreateGameBody(BaseModel):
    mode: str  # "public" | "private"
    ai: bool = False


def generate_room_code() -> str:
    import random, string
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))


@app.post('/api/games')
def create_game(body: CreateGameBody, user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    now = datetime.now(timezone.utc)
    doc = {
        "mode": body.mode,
        "ai": body.ai,
        "room_code": generate_room_code() if body.mode == "private" else None,
        "players": [user["_id"], None],
        "symbols": ["X", "O"],
        "moves": [],
        "board": [None]*9,
        "turn": "X",
        "result": None,
        "spectators": [],
        "created_at": now,
        "updated_at": now,
    }
    res = db["game"].insert_one(doc)
    game = db["game"].find_one({"_id": res.inserted_id})
    return serialize_doc(game)


@app.get('/api/games/public')
def list_public_games():
    games = db["game"].find({"mode": "public", "result": None}).sort("created_at", -1).limit(20)
    return [serialize_doc(g) for g in games]


@app.get('/api/games/{game_id}')
def get_game(game_id: str):
    game = db["game"].find_one({"_id": oid(game_id)})
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    return serialize_doc(game)


class JoinBody(BaseModel):
    room_code: Optional[str] = None


@app.post('/api/games/{game_id}/join')
def join_game(game_id: str, body: JoinBody, user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    game = db["game"].find_one({"_id": oid(game_id)})
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    if game.get("result"):
        raise HTTPException(status_code=400, detail="Game finished")
    if game.get("mode") == "private":
        if not body.room_code or body.room_code != game.get("room_code"):
            raise HTTPException(status_code=403, detail="Invalid room code")
    # Join as second player if slot empty and not already a player
    if user["_id"] not in game["players"]:
        if game["players"][1] is None:
            db["game"].update_one({"_id": game["_id"]}, {"$set": {"players": [game["players"][0], user["_id"]], "updated_at": datetime.now(timezone.utc)}})
        else:
            # Add as spectator
            db["game"].update_one({"_id": game["_id"]}, {"$addToSet": {"spectators": user["_id"]}, "$set": {"updated_at": datetime.now(timezone.utc)}})
    game = db["game"].find_one({"_id": game["_id"]})
    data = serialize_doc(game)
    # Notify room
    broadcast_game_update(str(game["_id"]), data)
    return data


class MoveBody(BaseModel):
    position: int


def symbol_for_user(game: dict, user: dict) -> Optional[str]:
    if game["players"][0] == user["_id"]:
        return "X"
    if game["players"][1] == user["_id"]:
        return "O"
    return None


@app.post('/api/games/{game_id}/move')
def make_move(game_id: str, body: MoveBody, user: dict = Depends(get_current_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    game = db["game"].find_one({"_id": oid(game_id)})
    if not game:
        raise HTTPException(status_code=404, detail="Game not found")
    if game.get("result"):
        raise HTTPException(status_code=400, detail="Game finished")
    pos = body.position
    if pos < 0 or pos > 8:
        raise HTTPException(status_code=400, detail="Invalid position")
    if game["board"][pos] is not None:
        raise HTTPException(status_code=400, detail="Cell occupied")
    user_symbol = symbol_for_user(game, user)
    # In AI games, if second player not set, human is X and AI is O
    if not user_symbol:
        raise HTTPException(status_code=403, detail="Not a player in this game")
    if user_symbol != game["turn"]:
        raise HTTPException(status_code=400, detail="Not your turn")

    # Apply human move
    board = game["board"]
    board[pos] = user_symbol
    moves = game["moves"] + [pos]
    result = check_winner(board)
    turn = 'O' if game["turn"] == 'X' else 'X'

    update = {"board": board, "moves": moves, "turn": turn, "updated_at": datetime.now(timezone.utc)}
    if result:
        update["result"] = result
    db["game"].update_one({"_id": game["_id"]}, {"$set": update})
    game = db["game"].find_one({"_id": game["_id"]})

    # If AI game and not finished and opponent is AI, make AI move
    if game.get("ai") and not game.get("result"):
        # If only one human joined, treat missing second player as AI with symbol 'O'
        ai_symbol = 'O' if game["players"][1] is None else ('X' if game["players"][0] is None else None)
        if ai_symbol == game["turn"]:
            ai_board = game["board"][:]
            ai_pos = best_ai_move(ai_board, ai_symbol)
            ai_board[ai_pos] = ai_symbol
            ai_moves = game["moves"] + [ai_pos]
            ai_result = check_winner(ai_board)
            next_turn = 'O' if ai_symbol == 'X' else 'X'
            update2 = {"board": ai_board, "moves": ai_moves, "turn": next_turn, "updated_at": datetime.now(timezone.utc)}
            if ai_result:
                update2["result"] = ai_result
            db["game"].update_one({"_id": game["_id"]}, {"$set": update2})
            game = db["game"].find_one({"_id": game["_id"]})

    data = serialize_doc(game)
    broadcast_game_update(str(game["_id"]), data)
    return data


@app.get('/api/users/{user_id}/games')
def user_games(user_id: str, user: dict = Depends(get_current_user)):
    if not user or str(user["_id"]) != user_id:
        raise HTTPException(status_code=401, detail="Not authorized")
    games = db["game"].find({"players": {"$in": [user["_id"]]}}).sort("created_at", -1).limit(50)
    return [serialize_doc(g) for g in games]


# ---------------------------
# WebSockets - simple room broadcaster
# ---------------------------
class ConnectionManager:
    def __init__(self):
        self.rooms: Dict[str, List[WebSocket]] = {}

    async def connect(self, room: str, websocket: WebSocket):
        await websocket.accept()
        self.rooms.setdefault(room, []).append(websocket)

    def disconnect(self, room: str, websocket: WebSocket):
        if room in self.rooms and websocket in self.rooms[room]:
            self.rooms[room].remove(websocket)
            if not self.rooms[room]:
                del self.rooms[room]

    async def broadcast(self, room: str, message: dict):
        if room not in self.rooms:
            return
        to_remove = []
        for ws in list(self.rooms[room]):
            try:
                await ws.send_json(message)
            except Exception:
                to_remove.append(ws)
        for ws in to_remove:
            self.disconnect(room, ws)


manager = ConnectionManager()


def broadcast_game_update(room: str, data: dict):
    import anyio
    # Fire-and-forget broadcast using anyio
    async def _send():
        await manager.broadcast(room, {"type": "game:update", "data": data})
    try:
        anyio.from_thread.run(_send)
    except Exception:
        pass


@app.websocket('/ws')
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    room: Optional[str] = None
    try:
        while True:
            msg = await websocket.receive_json()
            mtype = msg.get("type")
            if mtype == "join":
                gid = msg.get("gameId")
                if room:
                    manager.disconnect(room, websocket)
                room = str(gid)
                await manager.connect(room, websocket)
                # send initial state
                game = db["game"].find_one({"_id": oid(room)})
                if game:
                    await websocket.send_json({"type": "game:update", "data": serialize_doc(game)})
            elif mtype == "chat":
                if room:
                    await manager.broadcast(room, {"type": "chat", "data": {"text": msg.get("text", "")}})
    except WebSocketDisconnect:
        if room:
            manager.disconnect(room, websocket)


# ---------------------------
# Health & DB test
# ---------------------------
@app.get("/")
def read_root():
    return {"message": "TicTacToe API running"}


@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Not Connected",
        "collections": []
    }
    try:
        if db is not None:
            response["database"] = "✅ Available"
            try:
                collections = db.list_collection_names()
                response["collections"] = collections
                response["connection_status"] = "Connected"
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️ Connected but Error: {str(e)[:80]}"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:80]}"
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
