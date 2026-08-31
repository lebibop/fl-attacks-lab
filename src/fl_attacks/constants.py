class Datasets:
    MNIST = "mnist"
    FASHION_MNIST = "fashionmnist"
    CIFAR10 = "cifar10"
    CIFAR100 = "cifar100"


class Attacks:
    NONE = "none"
    LABEL_FLIP = "label_flip"
    SIGN_FLIP = "sign_flip"
    GAUSSIAN_NOISE = "gaussian_noise"
    BACKDOOR = "backdoor"