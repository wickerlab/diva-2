class BasePoisoner:
    def __init__(self, spacing, poisoning_rate):
        self.spacing = spacing
        self.poisoning_rate = poisoning_rate

    def poison(self, dataset):
        raise NotImplementedError("Subclasses must implement the poison method.")