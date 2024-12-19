## Gradient Descent Optimization Algorithms
**KurwAI Theoretical Article🦫**
<p align="center">
<img src="https://images.emojiterra.com/google/noto-emoji/unicode-15/color/512px/1f9ab.png" width=5%>
<img src="https://images.emojiterra.com/google/noto-emoji/unicode-16.0/color/svg/2764.svg" width=5%>
<img src="https://www.svgrepo.com/show/444064/legal-license-mit.svg" width=5%>
</p>

---
### Table of Contents:
- What's up with optimization?
- Basic Optimizers
- - Batch Gradient Descent
- - Mini-batch Gradient Descent
- - Stochastic Gradient Descent
- Smarter Optimizers
- - Momentum
- - Nesterov Accelerated Gradient
- - Adagrad
- - Adadelta
- - RMSprop
- - Adam
- - AdaMax
- - Nadam
- Conclusion

### Gradient Descent Optimization
Neural Networks are large mathematical models with gazillions of parameters, which can random at a glance.
Magic of backpropagation is getting feedback from outputs and tuning all these *knobs*, so they represent dependencies between input and output.
During training we do nothing, but learning transformations that can transform input into desired output.
Moving in the direction of negative gradient proved to be a clever way of tuning weights.
**Quick Reminder:**
Gradient Descent Optimization proposes taking a function that measures difference between desired and actual outputs.
This function can be plotted with a graph in R dimensional space where R-1 dimensions are our parameters and 1 dimension (it's convenient to think about it as of a z dimension on 3d plot).
Then we take partial derivative of the cost w.r.t each parameter and move in the negative direction of the gradient, because that's how we find minima of function (minima of cost function in our case).<br>
<img src="https://miro.medium.com/v2/resize:fit:800/1*G5H5_3SOWbDyI-tr2bVc-A.png" width=40%><br>
*Image taken from [here](https://www.google.com/url?sa=i&url=https%3A%2F%2Foztinasrin.medium.com%2Fcost-loss-function-in-machine-learning-a0ed21095f97&psig=AOvVaw0bA1Yptu6uuI-6neG4ldYF&ust=1734631946757000&source=images&cd=vfe&opi=89978449&ved=0CBcQjhxqFwoTCMjG9dr1sYoDFQAAAAAdAAAAABAJ)*
 <br>
However this is only one of plenty ways we can optimize our parameters.
Let's dig deeper!

### Basic Optimizers
#### Batch Gradient Descent
This is this very simple optimizer, you've probably worked with.
Its update rule is written in the following form:<br>
$$\theta_t = \theta_{t-1} - \alpha \nabla J(\theta_{t-1})$$
<br>
$t$ - timestep (people dealt with RNNs understand it  better)
$\alpha$ - learning rate
It does nothing, but moves in the scaled by $\alpha$ negative direction of gradient.
It's pretty popular in the world of machine learning.
Despite simplicity it's impractical for the following reasons:
1) It calculates cost on entire dataset. With many samples in data and layers in network it is really computationally inefficient .
2) It is stuck on plateus and local optimas (kinda rare tbh)
3) Sensitive to the scale of data. For example the cost function below (top-view) has a steep direction. That is to say our optimizer will perform a lot of meaningless updates.

<br>

<img src="https://i.ibb.co/9NjXCqh/2024-12-18-212850.png" width=40%><br>
*Image taken from [here](https://www.youtube.com/watch?v=tIovUOirJkE)*
 <br>
 
#### Mini-batch Gradient Descent
Mini-batch gradient descent update rule is similar to batch gradient descent one, but as a name implies we operate on parts of our datasets (mini-batches).
Update rule for Mini-batch gradient descent:
<br>
$$\theta_t = \theta_{t-1} - \alpha \nabla J(\theta_{t-1}, x^{(i:i+n)}, y^{(i:i+n)})$$
<br>
$n$ - batch size.
$n$ is typically a power of 2 (128, 256, 512 etc.)
Mini batch gradient descent is my fav among basic optimizers, because of its *temperance*.
It doesn't work with an entire dataset, but also it doesn't take 1 sample per update as the following optimizer does:

#### Stochastic Gradient Descent
As I spoilered above, stochastic gradient descent (SGD) takes only 1 sample for an update.
Even though it sounds crazy to calculate cost for only 1 sample (what if it's an outlier😨), SGD converges optima & updates our parameters wisely.
Update Rule:<br>
$$\theta_t = \theta_{t-1} - \alpha \nabla J(\theta_{t-1}, x^{(i)}, y^{(i)})$$
<br>
There's an interesting difference between convergence of Batch Gradient Descent and SGD:
<br>
<img src="https://statusneo.com/wp-content/uploads/2023/09/Credit-Research-Gate.jpg" width=40%><br>
*Image taken from [here](https://www.google.com/url?sa=i&url=https%3A%2F%2Fstatusneo.com%2Fefficient-opti-mastering-stochastic-gradient-descent%2F&psig=AOvVaw3iuWROPnvfsCKfCeyVTbhc&ust=1734633908012000&source=images&cd=vfe&opi=89978449&ved=0CBcQjhxqFwoTCMC534H9sYoDFQAAAAAdAAAAABAY)* <br>
As you can see SGD updates are way sharper, however they converge optimum, just as Batch Gradient Descent updates do.

---

### Smarter Optimizers
These are optimizers with special tricks that improve performance.
####  Momentum Gradient Descent
Momentum Optimizer is super cool. Remember that problem with narrow cost function bowl?
Momentum addresses this issue improving meaningful updates.
Useless updates are oscillations jumping in the opposite directions.
We'd better dampen them & that what momentum does!
Update rule:<br>
$$v_t = \beta v_{t-1} +\alpha \nabla J(\theta_{t-1})$$
$$\theta_t = \theta_{t-1} - v_t$$

Ok, let's clarify what's happening here.<br>
We have a variable $v$ (momentum term) that **accumulates**  previous updates scaled by $\beta$ (typically 0.9).
It gives us an interesting property of "forgetting" old updates & most importantly, acceleration in the relative direction & dampening of irrelevant updates (they are opposite directions (opposite signs), so it decreases oscillating updates.<br>
No momentum:<br>
<img src="https://people.willamette.edu/~gorr/classes/cs449/figs/valley1.gif" width=35%><br>
Momentum:<br>
<img src="https://people.willamette.edu/~gorr/classes/cs449/figs/valley2.gif" width=35%><br>
*Image taken from [here](https://people.willamette.edu/~gorr/classes/cs449/momrate.html)*

### Nesterov Accelerated Gradient (NAG) 
NAG follows Momentum optimizer idea improving it with **Look Ahead Term**.
Updates perform smarter, because optimizer looks forward.
It is a small adjustment to Momentum with a partial derivative of cost taken w.r.t to **approximation of update parameters**
Update rule:<br>
$$v_t = \beta v_{t-1} + \alpha \nabla J(\theta_{t-1} - \beta v_{t-1})$$
$$\theta_t = \theta_{t-1} -v_t$$
<br>
Easy and clever.<br>
<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ_-et8h51sUBJOeSTXIcv8Jb7RXliVD4yAVw&s" width=60%><br>
*Image taken from [here](https://www.google.com/url?sa=i&url=https%3A%2F%2Fpaperswithcode.com%2Fmethod%2Fnesterov-accelerated-gradient&psig=AOvVaw1bAH-zGwYX-uyKEj3o8dy0&ust=1734693737371000&source=images&cd=vfe&opi=89978449&ved=0CBcQjhxqFwoTCKi51fXbs4oDFQAAAAAdAAAAABAJ)*
Instead of adding accumulated gradients to gradient calculated at current timestep(blue vectors) we first of all, make a previously accumulated gradients jump(brown vector), then we calculate gradient at that point(red vector) and add them up (resulting green vector).<br>
Easy enough.

---

GUIDE IN DEVELOPMENT.
KURW.AI
