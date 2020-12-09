from flask import Flask, make_response
from flask import request
from flask import abort
from flask_cors import CORS
import Arena
from MCTS import MCTS
from othello.OthelloGame import OthelloGame
from othello.OthelloPlayers import *
from othello.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

game = OthelloGame(8)
ai = None
app = Flask(__name__)
CORS(app, supports_credentials=True)


@app.after_request
def after_request(resp):
    resp = make_response(resp)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST'
    resp.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return resp


def ai_factory():
    n = NNet(game)
    n.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts = MCTS(game, n, args)
    player = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
    print("model loaded")
    return player


def show_board(board):
    for i in range(-1, 8):
        print("%3d" % i, end='')
    print()
    for i in range(8):
        print("%3d" % i, end='')
        for j in range(8):
            print("%3d" % board[j][i], end='')
        print()


def action_to_position(action):
    return [action // 8, action % 8]


# 这个地方结尾得加 / 否则请求的时候如果结尾加了 / 就会 404
# 加了斜线之后如果请求没有 / 会自动 308
# 但是 308 之后前端又不好使了，真是太沙雕了，所以还是前端注意一下吧，这里不写了
@app.route('/prob', methods=['POST'])
def prob():
    body_json = request.json
    board = body_json['board']
    cur_player = body_json['cur_player']
    print(body_json)
    try:
        np_board = np.array(board, dtype=np.int)
        action = ai(game.getCanonicalForm(np_board, cur_player))
        return {"action": int(action)}
    except Exception:
        abort(500)


@app.route('/is_end', methods=['POST'])
def is_end():
    body_json = request.json
    board = body_json['board']
    cur_player = body_json['cur_player']
    try:
        np_board = np.array(board, dtype=np.int)
        game_end = game.getGameEnded(np_board, cur_player)
        return {"game_end": game_end}
    except Exception:
        abort(500)


@app.route('/next_state', methods=['POST'])
def next_state():
    body_json = request.json
    board = body_json['board']
    cur_player = body_json['cur_player']
    action = body_json['action']
    try:
        np_board = np.array(board, dtype=np.int)
        board, cur_player = game.getNextState(np_board, cur_player, action)
        return {"board": board.tolist(), "cur_player": int(cur_player)}
    except Exception:
        abort(500)


@app.route('/valid', methods=['POST'])
def valid():
    body_json = request.json
    board = body_json['board']
    cur_player = body_json['cur_player']
    try:
        np_board = np.array(board, dtype=np.int)
        valids = game.getValidMoves(game.getCanonicalForm(np_board, cur_player), 1)
        positions = np.argwhere(valids == 1).flatten().tolist()
        twoDPositions = [action_to_position(i) for i in positions]
        return {"valid_actions": twoDPositions}
    except Exception:
        abort(500)


def test():
    board = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0],
             [0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    show_board(board)
    board = np.array(board, dtype=np.int)
    test_ai = ai_factory()
    next_step = test_ai(board)
    print(next_step)
    print(action_to_position(26))
    next_board, next_player = game.getNextState(board, 1, 26)
    show_board(next_board)
    return next_step


if __name__ == '__main__':
    ai = ai_factory()
    app.run()
