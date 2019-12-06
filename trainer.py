from scripts.model import ResNext

my_model = ResNext(epochs=100, batchsize=6, learning_rate=1e-3, blocks=2, cardinality=4, depth=32)
my_model.net_init()
my_model.start_training(load_model=False, index=None)
