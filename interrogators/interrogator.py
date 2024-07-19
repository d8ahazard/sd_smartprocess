# Borrowed from https://github.com/toriato/stable-diffusion-webui-wd14-tagger/blob/master/tagger/interrogator.py
import importlib
import pkgutil
import re
import sys
from abc import abstractmethod

import torch
from PIL import Image

import extensions.sd_smartprocess.interrogators as interrogators
from extensions.sd_smartprocess.process_params import ProcessParams


@abstractmethod
class Interrogator:
    def __init__(self, params: ProcessParams) -> None:
        self.params = params
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        registry = InterrogatorRegistry()
        registry.register(self)

    def interrogate(self, image: Image, params: ProcessParams, unload: bool = False) -> str:
        raise NotImplementedError

    def unload(self):
        if self.model:
            try:
                self.model = self.model.to("cpu")
            except:
                pass

    def load(self):
        if self.model:
            try:
                self.model = self.model.to(self.device)
            except:
                pass


re_special = re.compile(r'([\\()])')


class InterrogatorRegistry:
    _instance = None  # Class variable to hold the singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InterrogatorRegistry, cls).__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self):
        self.interrogators = {}

    def register(self, interrogator: Interrogator):
        self.interrogators[interrogator.__class__.__name__] = interrogator

    def get_interrogator(self, interrogator_name: str) -> Interrogator:
        return self.interrogators[interrogator_name]

    def get_interrogators(self):
        return self.interrogators

    def unload(self):
        for interrogator in self.interrogators.values():
            interrogator.unload()

    def load(self):
        for interrogator in self.interrogators.values():
            interrogator.load()

    @staticmethod
    def list_interrogators(return_cls=False):
        # First, dynamically import all modules in the package to ensure all subclasses are loaded
        package = interrogators
        for importer, modname, ispkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
            print(f"Importing {modname}")
            try:
                importlib.import_module(modname)
            except Exception as e:
                print(f"Error importing {modname}: {e}")

        # Now, enumerate subclasses of Interrogator to access their class-level 'params'
        interrogator_dict = {}
        subclass_names = [cls.__name__ for cls in Interrogator.__subclasses__()]
        print(f"Subclass names: {subclass_names}")
        for cls in Interrogator.__subclasses__():
            if return_cls:
                interrogator_dict[cls.__name__] = cls
                continue
            params = getattr(cls, "params", None)
            if params is not None:  # Ensure 'params' is defined at the class level
                interrogator_dict[cls.__name__] = params
            else:
                interrogator_dict[cls.__name__] = {}

        return interrogator_dict

