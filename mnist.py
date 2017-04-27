import numpy as np
import os

import cntk as C
from cntk.io import MinibatchSource, CTFDeserializer, StreamDef, StreamDefs, INFINITELY_REPEAT
from cntk.layers import Sequential, Dense
from cntk.ops import input, element_times, constant, sigmoid
from cntk.losses import cross_entropy_with_softmax
from cntk.metrics import classification_error
from cntk.learners import sgd, learning_rate_schedule, UnitType
from cntk.logging import ProgressPrinter
from cntk.train import Trainer, training_session
from cntk.device import try_set_default_device, gpu

data_dir = abs_path = os.path.dirname(os.path.abspath(__file__)) + '/datasets/MNIST/'


def create_reader(path, is_training):
	return MinibatchSource(
		CTFDeserializer(
			path, 
			StreamDefs(
    			features  = StreamDef(field='features', shape=784, is_sparse=False),
    			labels    = StreamDef(field='labels',   shape=10,  is_sparse=False)
			)
		), 
		randomize = is_training, 
		max_sweeps = INFINITELY_REPEAT if is_training else 1)


def create_model(hidden_dim, output_dim, feature, label):
	scaled_feature = element_times(constant(0.00390625), feature)

	model = Sequential([
			Dense(hidden_dim, activation=sigmoid),
			Dense(output_dim)
		])

	return model(scaled_feature)


def train(path, hidden_dim, output_dim, feature, label):
	
	minibatch_size = 100
	num_inputs = 60000
	num_epochs = 10

	reader = create_reader(path, True)
	input_map = {
        feature  : reader.streams.features,
        label  : reader.streams.labels
    }

	y = create_model(hidden_dim, output_dim, feature, label)
	loss = cross_entropy_with_softmax(y, label)
	label_error = classification_error(y, label)

	lr = learning_rate_schedule(0.2, UnitType.minibatch)

	trainer = Trainer(
		model = y,
		criterion = (loss, label_error),
		parameter_learners = sgd(y.parameters, lr),
		progress_writers = [ProgressPrinter(tag='train', num_epochs=num_epochs)]
	)

	session = training_session(
        trainer = trainer,
        mb_source = reader,
        mb_size = minibatch_size,
        model_inputs_to_streams = input_map,
        max_samples = num_inputs * num_epochs,
        progress_frequency = num_inputs
    )

	session.train()
	
	return trainer


def test(path, trainer, feature, label):
	num_inputs = 10000
	minibatch_size = 1024

	num_minibatches_to_test = num_inputs // minibatch_size
	reader = create_reader(path, False)
	input_map = {
        feature  : reader.streams.features,
        label  : reader.streams.labels
    }

	test_result = 0.0

	for i in range(num_minibatches_to_test):
		minibatch = reader.next_minibatch(minibatch_size, input_map = input_map)
		eval_error = trainer.test_minibatch(minibatch)
		test_result = test_result + eval_error

	return test_result / num_minibatches_to_test


if __name__=='__main__':
	# try_set_default_device(gpu(0))

	train_data_path = data_dir + 'Train-28x28_cntk_text.txt'
	test_data_path = data_dir + 'Test-28x28_cntk_text.txt'

	input_dim = 784
	output_dim = 10

	feature = input(input_dim, np.float32)
	label = input(output_dim, np.float32)

	trainer = train(train_data_path, 15, 10, feature, label)

	error = test(test_data_path, trainer, feature, label)
	print("Error: %f" % error)
