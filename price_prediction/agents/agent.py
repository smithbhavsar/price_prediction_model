import os, re

from base import Base
from items import Item
from numexpr.utils import detect_number_of_cores

class Agent(Base):
    """
    An abstract superclass for Agents
    Used to log messages in a way that can identify each Agent
    """

    # Foreground colors
    RED = '\033[91m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Background color
    BG_BLACK = '\033[40m'
    
    # Reset code to return to default color
    RESET = '\033[0m'

    name: str = ""
    color: str = WHITE

    def __init__(self):
        super().__init__()

        ## numexpr emits logging if this isn't set
        cores = os.getenv("NUMEXPR_MAX_THREADS")
        if cores is None:
            cores = str(detect_number_of_cores())

        os.environ["NUMEXPR_MAX_THREADS"] = cores

    def initialize(self, message:str = ""):
        self.log(f"Initializing{message}")

    def ready(self):
        self.log("Ready!")


    def price(self, item:Item) -> float:
        prompt = item.prompt.replace('How much does this cost to the nearest dollar?\n\n', '')
        prompt = re.sub(r'\n+Price is \$[\d\.]+', '', prompt)

        return self._price(prompt)


    def log(self, message):
        """
        Log this as an info message, identifying the agent
        """
        self.logger.info(f'{self.BG_BLACK}{self.color}{message}{self.RESET}')
