from abc import abstractmethod


# base class for transformation
class transformation(object):  # pylint: disable=too-few-public-methods

    def __init__(self, params=None):
        """
        
        Args:
            params : 
        """
        self.params = params

    @abstractmethod
    def transform(self, X_prev):
        pass
