from scripts.model import ResNext

my_model = ResNext(epochs=100, batchsize=8, learning_rate=5e-4, blocks=2, cardinality=4, depth=32)
my_model.net_init()
my_model.start_training(load_model=False, index=None)
