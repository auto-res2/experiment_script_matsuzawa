"""
Configuration for DiffuSynerMix experiments.
"""

class ExperimentConfig:
    SEED = 42
    DEVICE = 'cuda'  # or 'cpu' if CUDA not available
    
    DATASET = 'cifar100'
    NUM_CLASSES = 100
    DATA_DIR = './data'
    
    BATCH_SIZE = 128
    NUM_WORKERS = 4
    NUM_EPOCHS = 20  # Set to 1 for testing
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4
    
    MODEL_NAME = 'resnet18'
    PRETRAINED = False
    SAVE_PATH = './models'
    
    MIXUP_ALPHA = 1.0
    
    DIFFU_HIDDEN_DIM = 64
    DIFFU_NUM_STEPS = 5
    DIFFU_NOISE_STD = 0.1
    
    EXPERIMENT_1_MODES = ['standard', 'synermix', 'diffusynermix']
    
    ABLATION_CONFIGS = {
        'no_direction': {'use_direction_predictor': False, 'num_steps': 5},
        'steps_3': {'use_direction_predictor': True, 'num_steps': 3},
        'steps_5': {'use_direction_predictor': True, 'num_steps': 5},
        'steps_7': {'use_direction_predictor': True, 'num_steps': 7}
    }
    
    TEST_MODE = True  # Set to False for full training
    
    @classmethod
    def get_test_config(cls):
        """Returns a copy of the config with minimal settings for testing."""
        test_config = ExperimentConfig()
        test_config.NUM_EPOCHS = 1
        test_config.BATCH_SIZE = 8
        test_config.NUM_WORKERS = 0
        test_config.EXPERIMENT_1_MODES = ['standard']
        test_config.ABLATION_CONFIGS = {
            'no_direction': {'use_direction_predictor': False, 'num_steps': 2}
        }
        return test_config
