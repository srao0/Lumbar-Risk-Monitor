from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import time

BoardShim.enable_dev_board_logger()

params = BrainFlowInputParams()
board = BoardShim(BoardIds.SYNTHETIC_BOARD.value, params)

board.prepare_session()
board.start_stream()

time.sleep(5)   # let data collect

data = board.get_board_data()

board.stop_stream()
board.release_session()

print(data.shape)
