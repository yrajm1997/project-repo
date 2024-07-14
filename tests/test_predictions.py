"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import numpy as np
import tensorflow as tf
from catvsdog_model import __version__ as _version
from catvsdog_model.config.core import config
from catvsdog_model.predict import make_prediction
from catvsdog_model.processing.data_manager import load_model


def test_make_prediction(sample_input_data):
    # Given
    for data, labels in sample_input_data:
        data_in = data[0]
        break

    # When
    data_in = tf.reshape(data_in, (1, 180, 180, 3))
    results = make_prediction(input_data = data_in)
    y_pred = results['predictions'][0]
    
    # Then
    assert y_pred is not None
    assert y_pred in ['cat', 'dog']
    assert results['version'] == _version


def test_accuracy(sample_input_data):
    # Given
    model_file_name = f"{config.app_config.model_save_file}{_version}"
    clf_model = load_model(file_name = model_file_name)
    
    # When
    test_loss, test_acc = clf_model.evaluate(sample_input_data, verbose=0)
    
    # Then
    assert test_acc > 0.6
