import time
import numpy as np

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams


board_id = 0
params = BrainFlowInputParams()
params.serial_port = "COM3"


board = BoardShim(board_id, params)
board.prepare_session()
board.start_stream(45000)
print("Start OpenBCI...")


def get_data():
    time.sleep(10)
    # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    data = board.get_board_data()  # get all data and remove it from internal buffer

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # noinspection PyTypeChecker
    np.savetxt("openbci_emotion_eeg/eeg_" + current_time + ".txt", data)
    return data


if __name__ == '__main__':
    get_data()
