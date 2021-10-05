import numpy as np
from typing import Collection, Tuple, Union

Contour = Collection[Tuple[Union[str, Tuple[Tuple[float, ...],
                                            Tuple[Tuple[float, ...]]]],
                           np.ndarray]]
