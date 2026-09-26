"""Background loader: only the accepted subfolders feed builds; ids are paths relative to the root."""

import cv2
import numpy as np
import pytest

from synthetic_backgrounds.loader import list_background_sources


def test_background_loader_scans_only_accepted_subfolders(tmp_path):
    for folder in ("flux", "photos/kitchen", "rejected", "procedural"):
        (tmp_path / folder).mkdir(parents=True)
        cv2.imwrite(str(tmp_path / folder / "image.jpg"), np.zeros((8, 8, 3), np.uint8))
    ids = [source.background_id for source in list_background_sources(tmp_path)]
    assert ids == ["flux/image.jpg", "photos/kitchen/image.jpg"]
    with pytest.raises(FileNotFoundError, match="no accepted background subfolder"):
        list_background_sources(tmp_path / "rejected")

