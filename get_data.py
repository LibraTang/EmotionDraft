import argparse
import time
import numpy as np
import shared_variable

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


board_id = 2
params = BrainFlowInputParams()
params.serial_port = "COM3"


def get_data():
    # board = BoardShim(board_id, params)
    # board.prepare_session()
    # # board.start_stream () # use this for default options
    # board.start_stream(45000)
    # time.sleep(10)
    # # data = board.get_current_board_data (256) # get latest 256 packages or less, doesnt remove them from internal buffer
    # data = board.get_board_data()  # get all data and remove it from internal buffer
    # board.stop_stream()
    # board.release_session()
    #
    # # print(data.shape)
    #
    # current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    # # noinspection PyTypeChecker
    # np.savetxt("openbci_emotion_eeg/eeg_" + current_time + ".txt", data)
    #
    # # 将收集的eeg添加到共享数组
    # np.append(shared_variable.data, data)
    return


if __name__ == '__main__':
    get_data()
