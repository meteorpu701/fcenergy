# src/fl/algorithms/__init__.py
from src.fl.algorithms.fedavg import FedAvg
from src.fl.algorithms.fednova import FedNova
from src.fl.algorithms.scaffold import Scaffold
from src.fl.algorithms.zeno import Zeno
from src.fl.algorithms.krum import Krum  

AGGREGATORS = {
    "fedavg": FedAvg,
    "fednova": FedNova,
    "scaffold": Scaffold,
    "zeno": Zeno,
    "krum": Krum,  
}