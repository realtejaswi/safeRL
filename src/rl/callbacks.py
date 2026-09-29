from stable_baselines3.common.callbacks import BaseCallback

class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            if "clearance" in info:
                self.logger.record("safety/clearance", info["clearance"])
            if "is_collision" in info:
                self.logger.record("safety/is_collision", float(info["is_collision"]))
        return True
