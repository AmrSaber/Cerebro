import argparse
import Cerebro.model.reader as reader

parser = argparse.ArgumentParser(
	description='Read, filter and separate the data to testing and training data (filtering is optional)'
)
parser.add_argument('-d', type=str, required=True, choices=reader.available_datasets)
parser.add_argument('-q', action='store_true', help='Quite mode, no output in terminal')
parser.add_argument('-f', action='store_true', help='Filter non-face images')
args = parser.parse_args()

reader.set_dataset(args.d)
reader.used_module.split_data(args.q, args.f)