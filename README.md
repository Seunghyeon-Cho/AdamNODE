# AdamNODEs: When Neural ODE Meets Adaptive Moment Estimation
Recent work by Xia et al. leveraged the continuous-limit of the classical momentum accelerated gradient descent and proposed heavy-ball neural ODEs. While this model offers computational efficiency and high utility over vanilla neural ODEs, this approach often causes the overshooting of internal dynamics, leading to unstable training of a model. Prior work addresses this issue by using ad-hoc approaches, e.g., bounding the internal dynamics using specific activation functions, but the resulting models do not satisfy the exact heavy-ball ODE. In this work, we propose adaptive momentum estimation neural ODEs (AdamNODEs) that adaptively control the acceleration of the classical momentum-based approach. We find that its adjoint states also satisfy AdamODE and do not require ad-hoc solutions that the prior work employs. In evaluation, we show that AdamNODEs achieve the lowest training loss and efficacy over existing neural ODEs. We also show that AdamNODEs have better training stability than classical momentum-based neural ODEs. This result sheds some light on adapting the techniques proposed in the optimization community to improving the training and inference of neural ODEs further.

# Usage

## CIFAR-10
~~~
python ./cifar10/main.py --model adamnode
~~~

## MNIST
~~~
python ./mnist/mnist_full_run.py
~~~

## Silverbox
~~~
python silverbox_init.py
~~~
