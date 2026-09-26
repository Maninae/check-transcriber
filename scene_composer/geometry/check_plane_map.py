"""The exact map from a check's pixels to the sheet plane, including paper deformation.

Forward map F (check pixels u -> plane pixels p), used for every label:
1. u / dpi gives check inches (s, t); the paper height there is z = h(s, t) (paper_deformation.py).
2. Arc compression slides (s, t) toward the part lying on the sheet.
3. A rigid placement (rotation, plane pixels per inch, center) puts that point on the plane.
4. Parallax: a point lifted by z, seen by a pinhole camera at height C above its nadir N,
   lands where its ray meets the sheet: p = N + (p_flat - N) * C / (C - z). So a lifted edge
   shifts outward from the nadir, and more the farther it sits from it.

The image is warped with the inverse: for each plane pixel p, solve F(u) = p by the fixed
point u <- A^-1 (p - D(u)), where A is the flat affine part and D = F - A is small and smooth.
The inverse is solved on a coarse grid and interpolated; `inverse_residual_px` measures it.

Camera model note: the scene camera is a plane homography with no explicit position, so the
nadir is approximated by the plane point under the photo center and C by a 65 degree field of
view over the photo's long side. Labels stay exact regardless, since image and labels share F.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from scene_composer.geometry.paper_deformation import PaperDeformation

FIXED_POINT_ITERATIONS = 10
INVERSE_GRID_STEP_PX = 4


@dataclass
class CheckPlaneMap:
    """Placement plus deformation of one check: check pixels <-> sheet-plane pixels."""

    check_width_px: int
    check_height_px: int
    dpi: float
    center_plane: tuple[float, float]      # plane pixels
    rotation_degrees: float                # counter-clockwise on screen
    plane_pixels_per_inch: float
    deformation: PaperDeformation
    nadir_plane: tuple[float, float]       # plane point straight below the camera
    camera_height_plane_px: float

    def linear_part(self) -> np.ndarray:
        """2x2 matrix taking check-pixel offsets to plane-pixel offsets (rotation and scale)."""
        radians = np.deg2rad(self.rotation_degrees)
        rotation = np.array([[np.cos(radians), np.sin(radians)], [-np.sin(radians), np.cos(radians)]])
        return rotation * self.plane_pixels_per_inch / self.dpi

    def flat_affine(self) -> np.ndarray:
        """2x3 affine for the undeformed check: plane = L u + b, with the check center on `center_plane`."""
        linear = self.linear_part()
        offset = np.array(self.center_plane) - linear @ [self.check_width_px / 2, self.check_height_px / 2]
        return np.hstack([linear, offset[:, None]])

    def check_to_plane(self, points_px: np.ndarray) -> np.ndarray:
        """F: (N, 2) check pixel coordinates -> (N, 2) plane pixel coordinates."""
        points_px = np.asarray(points_px, np.float64)
        s, t = points_px[:, 0] / self.dpi, points_px[:, 1] / self.dpi
        affine = self.flat_affine()
        if self.deformation.is_flat:
            return points_px @ affine[:, :2].T + affine[:, 2]
        height_px = self.deformation.height(s, t) * self.plane_pixels_per_inch
        shift_s, shift_t = self.deformation.arc_compression(s, t)
        compressed = np.stack([s + shift_s, t + shift_t], axis=1) * self.dpi
        flat = compressed @ affine[:, :2].T + affine[:, 2]
        nadir = np.array(self.nadir_plane)
        return nadir + (flat - nadir) * (self.camera_height_plane_px / (self.camera_height_plane_px - height_px))[:, None]

    def plane_to_check_points(self, points_plane: np.ndarray) -> np.ndarray:
        """F^-1 for scattered plane points by fixed-point iteration (check pixel coordinates)."""
        affine = self.flat_affine()
        inverse_linear = np.linalg.inv(affine[:, :2])
        target = np.asarray(points_plane, np.float64) - affine[:, 2]
        guess = target @ inverse_linear.T
        if self.deformation.is_flat:
            return guess
        for _ in range(FIXED_POINT_ITERATIONS):
            displacement = self.check_to_plane(guess) - (guess @ affine[:, :2].T + affine[:, 2])
            guess = (target - displacement) @ inverse_linear.T
        return guess

    def plane_region_to_check_maps(self, x0: int, y0: int, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        """Check pixel coordinates (map_x, map_y, float32) for every plane pixel of a region.

        Solved on a grid every INVERSE_GRID_STEP_PX pixels, then bilinearly upsampled; the map is
        smooth (creases are rounded), so interpolation error stays far below a pixel.
        """
        step = INVERSE_GRID_STEP_PX
        coarse_x = np.arange(0, width + step, step, dtype=np.float64)
        coarse_y = np.arange(0, height + step, step, dtype=np.float64)
        grid_x, grid_y = np.meshgrid(coarse_x + x0, coarse_y + y0)
        solved = self.plane_to_check_points(np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)).reshape(*grid_x.shape, 2)
        fine = align_corners_upsample(solved, step)
        return np.ascontiguousarray(fine[:height, :width, 0]), np.ascontiguousarray(fine[:height, :width, 1])


def align_corners_upsample(coarse: np.ndarray, step: int) -> np.ndarray:
    """Bilinear upsampling where coarse node (i, j) sits on fine pixel (i * step, j * step)."""
    coarse_height, coarse_width = coarse.shape[:2]
    fine_height, fine_width = (coarse_height - 1) * step + 1, (coarse_width - 1) * step + 1
    fine_y, fine_x = np.mgrid[0:fine_height, 0:fine_width].astype(np.float32)
    return cv2.remap(coarse.astype(np.float32), fine_x / step, fine_y / step, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def inverse_residual_px(plane_map: CheckPlaneMap, x0: int, y0: int, width: int, height: int) -> float:
    """Max |F(F^-1(p)) - p| in plane pixels over region pixels whose preimage lies on the check."""
    map_x, map_y = plane_map.plane_region_to_check_maps(x0, y0, width, height)
    on_check = (map_x >= 0) & (map_x <= plane_map.check_width_px) & (map_y >= 0) & (map_y <= plane_map.check_height_px)
    rows, columns = np.nonzero(on_check)
    if len(rows) == 0:
        return 0.0
    preimages = np.stack([map_x[rows, columns], map_y[rows, columns]], axis=1)
    targets = np.stack([columns + x0, rows + y0], axis=1).astype(np.float64)
    return float(np.abs(plane_map.check_to_plane(preimages) - targets).max())
