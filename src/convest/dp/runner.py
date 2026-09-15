"""Offline training has no robot rollout; metrics come from the dataset."""
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner


class OfflineImageRunner(BaseImageRunner):
    def run(self, policy):
        return {}
