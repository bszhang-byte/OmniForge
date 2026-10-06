from torch.utils.data import Dataset as TorchDataset
from collections.abc import Iterator
from typing import Any
import numpy as np

class TransformableDataset(TorchDataset):
    """Marker base class for all PSI datasets.

    All PSI datasets must apply transforms internally so that data is
    fully transformed on access. Downstream APIs should type-hint against
    this class instead of individual dataset types.
    """

    def iter_episode(
        self,
        eps_idx: int | None = None,
        stride: int = 30,
        dataset_index: int | None = None,
    ) -> Iterator[Any]:
        """Yield transformed steps for one episode, advancing ``stride`` frames.

        ``dataset_index`` is required for unambiguous mixture evaluation because
        LeRobot episode IDs are local to each pack and commonly overlap.
        """
        raw_dataset = getattr(self, "raw_dataset", None)
        transform = getattr(self, "transform", None)
        transform_kwargs = getattr(self, "transform_kwargs", {})

        assert raw_dataset is not None, "raw_dataset is not set on this dataset"

        # Mode 2: ShardedLeRobotMixtureDataset
        if hasattr(raw_dataset, "iter_episode"):
            selected_datasets = (
                [raw_dataset.datasets[dataset_index]]
                if dataset_index is not None
                else raw_dataset.datasets
            )
            all_episode_ids: list[int] = [
                tid
                for ds in selected_datasets
                for shard in ds.sharded_trajectories
                for tid in shard
            ]
            total_episodes = len(all_episode_ids)
            if eps_idx is None:
                eps_idx = int(all_episode_ids[np.random.randint(0, total_episodes)])
            print(f"Episode {eps_idx} / {total_episodes} total")
            for i, frame in enumerate(
                raw_dataset.iter_episode(eps_idx, dataset_index=dataset_index)
            ):
                if i % stride == 0: # TODO optimize by skipping frames in the raw dataset instead of post-hoc
                    yield transform(frame, **transform_kwargs) if transform is not None else frame

        # Mode 1: plain LeRobot dataset
        elif hasattr(raw_dataset, "meta") and hasattr(raw_dataset.meta, "total_episodes"):
            total_episodes = raw_dataset.meta.total_episodes
            if eps_idx is None:
                eps_idx = np.random.randint(0, total_episodes)
            start_frame = raw_dataset.base_dataset.episode_data_index["from"][eps_idx].item()
            end_frame   = raw_dataset.base_dataset.episode_data_index["to"][eps_idx].item()
            print(f"Episode {eps_idx} / {total_episodes}, frames {start_frame}..{end_frame} ({end_frame - start_frame} frames)")
            for idx in range(start_frame, end_frame, stride):
                raw = raw_dataset[idx]
                yield transform(raw, **transform_kwargs) if transform is not None else raw

        else:
            raise NotImplementedError(
                f"iter_episode is not supported for raw_dataset type {type(raw_dataset).__name__}"
            )
