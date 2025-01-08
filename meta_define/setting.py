from abc import ABC, abstractmethod


class Setting(ABC):
    order = 0

    @abstractmethod
    def get_path(self) -> str:
        pass

    @abstractmethod
    def interpret(self, content, context: dict):
        pass
