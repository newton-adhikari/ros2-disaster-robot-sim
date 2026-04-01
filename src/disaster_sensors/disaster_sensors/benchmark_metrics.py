#!/usr/bin/env python3

import numpy as np
import argparse
import PIL.Image

# defined constants
BUILDING_AREA_M2  = 400.0     # the interior of the world is 20m × 20m interior
MAP_RESOLUTION    = 0.05      # SLAM Toolbox default
FREE_THRESH       = 220       # pixel value > 220 → free  (occ prob < 0.25)
OCCUPIED_THRESH   = 50        # pixel value < 50  → occupied


class BenchmarkMetrics:
    BENCHMARK_VERSION = "1.0"

    def __init__(
        self,
        building_area_m2: float = BUILDING_AREA_M2,
        map_resolution: float    = MAP_RESOLUTION,
        verbose: bool            = True
    ):
        self.building_area = building_area_m2
        self.resolution    = map_resolution
        self.verbose       = verbose


    @staticmethod
    def _read_pgm_fallback(path: str) -> np.ndarray:
        pass

    def compute_coverage(self, pgm_path: str) -> dict:
        # this computes the coverage of exploration from pgm of the grid
        try:
            img = np.array(PIL.Image.open(pgm_path).convert('L'))
        except Exception:
            # img = self._read_pgm_fallback(pgm_path)
            pass

        free_mask     = img > FREE_THRESH
        occupied_mask = img < OCCUPIED_THRESH
        unknown_mask  = ~free_mask & ~occupied_mask

        free_cells  = int(np.sum(free_mask))
        total_cells = img.size

        # Coverage: only count cells that are inside the building footprint.
        # We approximate the building interior as the full grid minus a 1-cell
        # border (exterior walls are occupied and will not contribute anyway).
        free_area_m2  = free_cells * (self.resolution ** 2)
        coverage_pct  = min(100.0, free_area_m2 / self.building_area * 100.0)

        result = {
            "coverage_pct":    round(coverage_pct, 2),
            "free_cells":      free_cells,
            "free_area_m2":    round(free_area_m2, 2),
            "occupied_cells":  int(np.sum(occupied_mask)),
            "unknown_cells":   int(np.sum(unknown_mask)),
            "total_cells":     total_cells,
            "map_path":        str(pgm_path)
        } 

        if self.verbose:
            print(f"  Coverage:  {coverage_pct:.1f}%  "
                  f"({free_cells} free cells, {free_area_m2:.1f} m²)")
        return result


def main():
    pass

if __name__ == '__main__':
    main()