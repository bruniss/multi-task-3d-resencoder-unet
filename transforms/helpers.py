import random

class RandomTransformChoice:
    """
    A wrapper that randomly chooses exactly one transform from a list of transforms,
    using optional weights.
    """
    def __init__(self, transforms, probabilities=None):
        """
        Args:
            transforms (list): A list of transform callables, e.g. instances of
                RandomFlipWithNormals, RandomRotate90WithNormals, etc.
            probabilities (list, optional): If provided, a list of probabilities
                (weights) that sum to 1.0, one per transform. Otherwise, uniform.
        """
        self.transforms = transforms
        # If probabilities are given, verify they match the number of transforms
        if probabilities is not None:
            if len(probabilities) != len(transforms):
                raise ValueError(
                    f"Expected {len(transforms)} probabilities, got {len(probabilities)}."
                )
            if not abs(sum(probabilities) - 1.0) < 1e-7:
                raise ValueError("Probabilities must sum to 1.0.")
        self.probabilities = probabilities

    def __call__(self, data_dict):
        """
        Randomly pick one transform from `self.transforms` (with `self.probabilities` if provided),
        then apply it to `data_dict`.
        """
        chosen_transform = random.choices(self.transforms, weights=self.probabilities, k=1)[0]
        return chosen_transform(data_dict)