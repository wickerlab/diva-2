from abc import abstractmethod

class BasePoisoner:
    def __init__(self, poisoning_rate):
        self.poisoning_rate = poisoning_rate

    @abstractmethod
    def poison(self, dataset):
        pass
