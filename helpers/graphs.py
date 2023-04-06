import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import *


from ipywidgets import widgets, interactive


def make_graph(actuals, predictions):
    assert len(actuals) == len(predictions), "len of actuals and predictions needs to be the same"


    plt.figure()
    x = range(len(actuals))
    plt.plot(x, actuals, ".", "r", label = "actuals")
    plt.plot(x, predictions, ".", "b", label = "predictions")

    plt.legend()
    plt.tight_layout()
    plt.show()

    
    

