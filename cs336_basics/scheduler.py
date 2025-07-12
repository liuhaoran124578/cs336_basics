import numpy as np


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    """
    Given the parameters of a cosine learning rate decay schedule (with linear
    warmup) and an iteration number, return the learning rate at the given
    iteration under the specified schedule.

    Args:
        it (int): Iteration number to get learning rate for.
        max_learning_rate (float): alpha_max, the maximum learning rate for
            cosine learning rate schedule (with warmup).
        min_learning_rate (float): alpha_min, the minimum / final learning rate for
            the cosine learning rate schedule (with warmup).
        warmup_iters (int): T_w, the number of iterations to linearly warm-up
            the learning rate.
        cosine_cycle_iters (int): T_c, the number of cosine annealing iterations.

    Returns:
        Learning rate at the given iteration under the specified schedule.
    """
    if it < warmup_iters:
        # Warm-up 阶段：线性增加学习率
        lr = (it / warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        # Cosine Annealing 阶段：余弦函数衰减
        t = it - warmup_iters
        T = cosine_cycle_iters - warmup_iters
        cos_value = np.cos(np.pi * t / T)
        lr = min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + cos_value)
    else:
        # Post-annealing 阶段：学习率保持最小值
        lr = min_learning_rate

    return lr
