import threading
import queue
from typing import (
    Dict,
    Any,
    Tuple,
    Optional,
    List
)
import time
from threading import Lock
import numpy as np
from pathlib import Path





from real_time_vital_analyze.base.producer_consumer_manager import ProducerConsumerManager
from real_time_vital_analyze.base.consumer_tool_pool import ConsumerToolPool
from real_time_vital_analyze.socket_server import SocketServer
from real_time_vital_analyze.lockfree_queue import LockFreeQueue, AtomicDict, AtomicLong

from real_time_vital_analyze.fixed_size_queue import FixedSizeSlidingQueue, FixedSizeAtomicDict
from neural_network.rnn.lstm_engine import LSTMEngine
from neural_network.rnn.auto_encoders import RecurrentAutoencoder
from real_time_vital_analyze.peak_state import RealTimeStateMonitor
from real_time_vital_analyze.sleep_data_state_storage import SleepDataStateStorage
from real_time_vital_analyze.provider.sql_provider import SqlProvider
from real_time_vital_analyze.tables.sleep_data_state import SleepDataState
from real_time_vital_analyze.tables.real_time_vital_data import RealTimeVitalData

SUB_ROOT_DIRECTORY = Path(__file__).parent
ROOT_DIRECTORY = Path(__file__).parent.parent
LSTM_MODEL_PATH = str(ROOT_DIRECTORY / "models" / "lstm" / "checkpoint_epoch_32_2dimensions_20000_no_normalized.pth")
SCALER_PATH = str(ROOT_DIRECTORY / "models" / "lstm" / "sleep_scaler_20W_2dimensions.pkl")
SQL_CONFIG_PATH = str(SUB_ROOT_DIRECTORY / "config" / "yaml" / "sql_config.yaml")



def test_lstm_engine():
    model_params = {
        'seq_len': 20,
        'n_features': 2,
        'embedding_dim': 128
    }


    try:
        engine = LSTMEngine(
            model_class=RecurrentAutoencoder,
            model_params=model_params,
            seq_len=20,
            n_features=2,
            threshold=5,
            normalized_flag=0
        )
        engine.setup(model_path=LSTM_MODEL_PATH, scaler_path=SCALER_PATH)
    except Exception as e:
        raise ValueError(f"fail to load the lstm engine! {str(e)}") from e


if __name__ == '__main__':
    from real_time_vital_analyze.tables.device_data import DeviceData
    sql_provider = SqlProvider(model=DeviceData, sql_config_path=SQL_CONFIG_PATH)
            
    existing_devices = sql_provider.get_record_by_condition(
        condition={"device_code": "13271C9D10004071111715B507"}
    )    
    print(existing_devices)