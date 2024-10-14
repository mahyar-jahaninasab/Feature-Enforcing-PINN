Citation: Mahyar Jahani-nasab, Bijarchi Mohammad Ali, "Enhancing convergence speed with feature enforcing physics-informed neural networks using boundary conditions as prior knowledge." Sci Rep 14, 23836 (2024).


# Enhancing Convergence Speed with Feature-Enforcing Physics-Informed Neural Networks: Utilizing Boundary Conditions as Prior Knowledge for Faster Convergence

This repository contains the implementation of a novel accelerated training method for Vanilla Physics-Informed-Neural-Networks (PINN). The method addresses three factors that imbalance the loss function: 

1. Initial weight state of a neural network
2. Domain to boundary points ratio
3. Loss weighting factor
## Abstract:
This study introduces an accelerated training method for Vanilla Physics-Informed-Neural-Networks (PINN) addressing three factors that imbalance the loss function: initial weight state of a neural network, domain to boundary points ratio, and loss weighting factor. We propose a novel two-stage training method. During the initial stage, we create a unique loss function using a subset of boundary conditions and partial differential equation terms. Furthermore, we introduce preprocessing procedures that aim to decrease the variance during initialization and choose domain points according to the initial weight state of various neural networks. The second phase resembles Vanilla-PINN training, but a portion of the random weights are substituted with weights from the first phase. This implies that the neural network's structure is designed to prioritize the boundary conditions, subsequently affecting the overall convergence. Three benchmarks are utilized: two-dimensional flow over a cylinder, an inverse problem of inlet velocity determination, and the Burger equation. It is found that incorporating weights generated in the first training phase into the structure of a neural network neutralizes the effects of imbalance factors. For instance, in the first benchmark, as a result of our process, the second phase of training is balanced across a wide range of ratios and is not affected by the initial state of weights, while the Vanilla-PINN failed to converge in most cases. Lastly, the initial training process not only eliminates the need for hyperparameter tuning to balance the loss function, but it also outperforms the Vanilla-PINN in terms of speed.

## Two-Stage Training Method

We propose a two-stage training method:

1. **Initial Stage**: A unique loss function is created using a subset of boundary conditions and partial differential equation terms. We introduce preprocessing procedures that aim to decrease the variance during initialization and choose domain points according to the initial weight state of various neural networks.

2. **Second Phase**: This phase resembles Vanilla-PINN training, but a portion of the random weights are substituted with weights from the first phase. This implies that the neural network's structure is designed to prioritize the boundary conditions, subsequently affecting the overall convergence.

## Benchmarks

Three benchmarks are utilized:

1. Two-dimensional flow over a cylinder
2. An inverse problem of inlet velocity determination
3. The Burger equation

Incorporating weights generated in the first training phase into the structure of a neural network neutralizes the effects of imbalance factors. For instance, in the first benchmark, as a result of our process, the second phase of training is balanced across a wide range of ratios and is not affected by the initial state of weights, while the Vanilla-PINN failed to converge in most cases.

## Contact

If you have any questions or need further clarification, feel free to reach out to me. You can email me at:
mahyarjahaninasab [at] gmail [dot] com








