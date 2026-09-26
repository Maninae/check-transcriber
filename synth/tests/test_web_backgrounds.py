"""Offline tests for the web-background fetcher: the license gate and perceptual dedupe."""

import cv2
import numpy as np
import pytest

from synth.backgrounds.fetch_web_backgrounds import process_candidate
from synth.backgrounds.perceptual_duplicate_index import PerceptualDuplicateIndex, band_correlations, perceptual_signature
from synth.backgrounds.procedural import write_procedural_backgrounds
from synth.backgrounds.web_candidate import (CC0_LICENSE, PUBLIC_DOMAIN, PUBLIC_DOMAIN_MARK, SurfaceCategory,
                                             WebBackgroundCandidate, WebBackgroundSource, accepted_license)
from synth.backgrounds.web_sources_commons import commons_page_to_candidate
from synth.backgrounds.web_sources_openverse import openverse_result_to_candidate


def openverse_result(license_code: str) -> dict:
    """A trimmed Openverse search result as the API returns it."""
    return {"id": "8dce4c32-159c-4198-abbe-116dcf7b8a45", "title": "Bed Sheets", "creator": "Someone",
            "license": license_code, "url": "https://example.org/sheets.jpg",
            "foreign_landing_url": "https://stocksnap.io/photo/x", "width": 3089, "height": 2048}


def commons_page(license_code: str, title: str = "File:Crumpled duvet cover.jpg", width: int = 4000) -> dict:
    """A trimmed Commons generator=search page with imageinfo + extmetadata."""
    return {"pageid": 123, "title": title, "index": 1, "imageinfo": [{
        "width": width, "height": 3000, "mime": "image/jpeg", "url": "https://upload.example/orig.jpg",
        "thumburl": "https://upload.example/2048px.jpg", "descriptionurl": "https://commons.example/File:x",
        "extmetadata": {"License": {"value": license_code}, "Artist": {"value": "<a href='x'>Photographer</a>"}},
    }]}


@pytest.mark.parametrize("license_code, expected", [
    ("cc0", CC0_LICENSE), ("pdm", PUBLIC_DOMAIN_MARK), ("by", None), ("by-sa", None), ("by-nc", None), ("", None),
])
def test_openverse_license_gate(license_code, expected):
    candidate = openverse_result_to_candidate(openverse_result(license_code), SurfaceCategory.BEDDING)
    assert accepted_license(candidate) == expected


@pytest.mark.parametrize("license_code, expected", [
    ("cc0", CC0_LICENSE), ("pd", PUBLIC_DOMAIN), ("cc-by-sa-4.0", None), ("cc-by-4.0", None), ("", None),
])
def test_commons_license_gate_reads_each_files_own_metadata(license_code, expected):
    candidate = commons_page_to_candidate(commons_page(license_code), SurfaceCategory.BEDDING)
    assert accepted_license(candidate) == expected
    assert candidate.author == "Photographer"
    assert candidate.image_url.endswith("2048px.jpg")


def test_license_codes_are_not_shared_across_sources():
    # "pdm" is an Openverse code; Commons never reports it, so it must not pass there.
    candidate = commons_page_to_candidate(commons_page("pdm"), SurfaceCategory.FABRIC)
    assert accepted_license(candidate) is None


def test_metadata_rules_drop_small_images_and_mirrors_before_download():
    assert commons_page_to_candidate(commons_page("cc0", width=800), SurfaceCategory.FABRIC) is None
    mirror_title = "File:Dirty carpet diff 8k (Rohit Seervi via Poly Haven).jpg"
    assert commons_page_to_candidate(commons_page("cc0", title=mirror_title), SurfaceCategory.RUG_CARPET) is None


class NoNetworkClient:
    """Fails the test if anything tries to download."""

    def get_bytes(self, url, extra_headers=None):
        raise AssertionError(f"downloaded {url} despite a failing license")


def test_rejected_license_is_never_downloaded(tmp_path):
    candidate = WebBackgroundCandidate(
        source=WebBackgroundSource.OPENVERSE, source_id="abc", slug="sheets", title="Sheets", author="x",
        source_license_code="by", license_evidence="fixture", source_page_url="", image_url="https://example.org/a.jpg",
        category=SurfaceCategory.BEDDING)
    row, reason = process_candidate(NoNetworkClient(), candidate, tmp_path, PerceptualDuplicateIndex(), "2026-09-25")
    assert row is None and reason.startswith("license")
    assert not list(tmp_path.iterdir())


@pytest.fixture(scope="module")
def two_fabrics(tmp_path_factory):
    """Two different procedural fabric backgrounds as uint8 RGB."""
    directory = tmp_path_factory.mktemp("fabrics")
    paths = write_procedural_backgrounds(directory, 2, seed=11)
    return [cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB) for path in paths]


def degraded_copy(image_rgb: np.ndarray, scale: float = 0.5, jpeg_quality: int = 70) -> np.ndarray:
    """What a second source's copy of the same photo looks like: smaller and recompressed."""
    resized = cv2.resize(image_rgb, (int(image_rgb.shape[1] * scale), int(image_rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    _, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)


def test_signature_survives_resize_and_jpeg(two_fabrics):
    coarse, fine = band_correlations(perceptual_signature(two_fabrics[0]), perceptual_signature(degraded_copy(two_fabrics[0])))
    assert coarse > 0.9 and fine > 0.7


def test_index_flags_duplicates_and_keeps_distinct_images(two_fabrics):
    index = PerceptualDuplicateIndex()
    index.add(perceptual_signature(two_fabrics[0]), "flux/a.png")
    brighter_copy = np.clip(degraded_copy(two_fabrics[0]).astype(np.int16) + 12, 0, 255).astype(np.uint8)
    assert index.find_duplicate(perceptual_signature(brighter_copy)) == "flux/a.png"
    assert index.find_duplicate(perceptual_signature(two_fabrics[1])) is None


def test_same_lighting_different_texture_is_not_a_duplicate(two_fabrics):
    # Both fabrics under one strong vignette: the coarse band agrees, the fine band must veto.
    height, width = two_fabrics[0].shape[:2]
    rows, columns = np.mgrid[0:height, 0:width]
    # Smooth light falloff to 50% in the corners (never clipped, so it adds no edge of its own).
    vignette = (1.0 - (((rows - height / 2) / height) ** 2 + ((columns - width / 2) / width) ** 2))[..., None]
    first, second = ((fabric * vignette).clip(0, 255).astype(np.uint8) for fabric in two_fabrics)
    coarse, fine = band_correlations(perceptual_signature(first), perceptual_signature(second))
    assert coarse > 0.85, "fixture should share lighting"
    index = PerceptualDuplicateIndex()
    index.add(perceptual_signature(first), "a")
    assert index.find_duplicate(perceptual_signature(second)) is None
