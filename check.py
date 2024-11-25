from cnnDigitReco.config.configuration import DataTransformationConfig

def train(config: DataTransformationConfig):
        config = config
        print(config.get_training_config(config))

train()