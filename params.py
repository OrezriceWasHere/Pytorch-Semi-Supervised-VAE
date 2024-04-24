
class Params:
    SAMPLE_SAPCE = [64, 7, 7]
    SAMPLE_SPACE_FLATTEN = 64 * 7 * 7
    LATENT_DIM = 100
    ENCODER_CONVOLUTIONS = [1, 32, 4, 1, 2, 32, 32, 4, 2, 1, 32, 64, 4, 2, 1]
    DECODER_CONVOLUTIONS = [64, 32, 4, 2, 1, 32, 32, 4, 2, 1, 1 ,32,  1, 4, 1, 2]
