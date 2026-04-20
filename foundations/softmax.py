import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)

        z = z - np.max(z)

        # np.exp() can handle arrays
        exp_array = np.exp(z)

        return np.round(exp_array / np.sum(exp_array), 4)
