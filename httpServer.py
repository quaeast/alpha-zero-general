from flask import Flask
from flask import request
from flask import abort
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


def ai_factory():
    n = NNet(game)
    n.load_checkpoint('./pretrained_models/othello/pytorch/', '8x8_100checkpoints_best.pth.tar')
    args = dotdict({'numMCTSSims': 50, 'cpuct': 1.0})
    mcts = MCTS(game, n, args)
    player = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
    print("model loaded")
    return player


def show_board(board):
    for line in board:
        for point in line:
            print("%3d" % point, end='')
        print()


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
    show_board(board)
    try:
        np_board = np.array(board, dtype=np.int)
        valids = game.getValidMoves(game.getCanonicalForm(np_board, cur_player), 1)
        positions = np.argwhere(valids == 1).flatten()
        return {"valids": positions.tolist()}
    except Exception:
        abort(500)


def test():
    board = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, -1, 1, 0, 0, 0],
             [0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    board = np.array(board, dtype=np.int)
    test_ai = ai_factory()
    next_step = test_ai(board)
    return next_step


if __name__ == '__main__':
    ai = ai_factory()
    app.run()
