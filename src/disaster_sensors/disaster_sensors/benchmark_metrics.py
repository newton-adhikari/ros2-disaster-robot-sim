#!/usr/bin/env python3

import numpy as np
import argparse
import PIL.Image
import csv


# defined constants
BUILDING_AREA_M2  = 400.0     # the interior of the world is 20m × 20m interior
MAP_RESOLUTION    = 0.05      # SLAM Toolbox default
FREE_THRESH       = 220       # pixel value > 220 → free  (occ prob < 0.25)
OCCUPIED_THRESH   = 50        # pixel value < 50  → occupied
COLLISION_RANGE_M = 0.30      # m — front-arc LiDAR < this → near-collision
COLLISION_ARC_DEG = 25        # ±25° frontal arc
COLLISION_MIN_DUR = 0.10      # s — minimum duration to count as event


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
        # this method is supposed to Read PGM without PIL
        # but this handles p5 binary format only
        with open(path, 'rb') as f:
            header = []
            while len(header) < 3:
                line = f.readline().decode('ascii').strip()
                if not line.startswith('#'):
                    header.extend(line.split())
            assert header[0] == 'P5', "Only binary PGM (P5) supported"
            width, height = int(header[1]), int(header[2])
            maxval = int(header[3]) if len(header) > 3 else int(
                f.readline().decode('ascii').strip())
            data = np.frombuffer(f.read(), dtype=np.uint8
                                 if maxval < 256 else np.uint16)
        return data.reshape(height, width).astype(np.float32)

    def compute_coverage(self, pgm_path: str) -> dict:
        # this computes the coverage of exploration from pgm of the grid
        try:
            img = np.array(PIL.Image.open(pgm_path).convert('L'))
        except Exception:
            img = self._read_pgm_fallback(pgm_path)

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
    
    
    def compute_localisation_rmse(self, ekf_csv_path: str) -> dict:
        # this method Computes localisation RMSE rho (m) from EKF log CSV
        # this expectes a csv precomputed
        
        timestamps, ekf_xs, ekf_ys, gt_xs, gt_ys = [], [], [], [], []

        with open(ekf_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    timestamps.append(float(row['timestamp']))
                    ekf_xs.append(float(row['ekf_x']))
                    ekf_ys.append(float(row['ekf_y']))
                    gt_xs.append(float(row['gt_x']))
                    gt_ys.append(float(row['gt_y']))
                except (KeyError, ValueError):
                    continue

        if len(timestamps) < 2:
            raise ValueError(f"Insufficient data in {ekf_csv_path}: "
                             f"only {len(timestamps)} rows")
        
        ekf_x = np.array(ekf_xs)
        ekf_y = np.array(ekf_ys)
        gt_x  = np.array(gt_xs)
        gt_y  = np.array(gt_ys)

        sq_errors = (ekf_x - gt_x)**2 + (ekf_y - gt_y)**2
        rmse      = float(np.sqrt(np.mean(sq_errors)))
        max_error = float(np.sqrt(np.max(sq_errors)))
        duration  = float(timestamps[-1] - timestamps[0])

        result = {
            "rmse_m":      round(rmse, 4),
            "max_error_m": round(max_error, 4),
            "n_samples":   len(timestamps),
            "duration_s":  round(duration, 1),
        }
        if self.verbose:
            print(f"  RMSE:      {rmse:.3f} m  (max {max_error:.3f} m, "
                  f"n={len(timestamps)}, T={duration:.0f}s)")
        return result
    
    def compute_near_collision_rate(self, collision_csv_path: str, trial_duration_s: float) -> dict:
        # this method  Compute near-collision rate nu
        rows = []
        with open(collision_csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    rows.append((
                        float(row['timestamp']),
                        float(row['min_front_range_m'])
                    ))
                except (KeyError, ValueError):
                    continue

        if not rows:
            return {"events": 0, "rate_per_min": 0.0,
                    "total_below_threshold_s": 0.0}
        

        events = 0
        in_event = False
        event_start = 0.0
        total_below = 0.0

        for ts, r in rows:
            if r < COLLISION_RANGE_M:
                if not in_event:
                    in_event = True
                    event_start = ts
            else:
                if in_event:
                    duration = ts - event_start
                    total_below += duration
                    if duration >= COLLISION_MIN_DUR:
                        events += 1
                    in_event = False

        if in_event:
            duration = rows[-1][0] - event_start
            total_below += duration
            if duration >= COLLISION_MIN_DUR:
                events += 1

        duration_min = trial_duration_s / 60.0
        rate_per_min = events / duration_min if duration_min > 0 else 0.0

        result = {
            "events":                  events,
            "rate_per_min":            round(rate_per_min, 3),
            "total_below_threshold_s": round(total_below, 1),
        }

        if self.verbose:
            print(f"  Near-coll: {events} events  "
                  f"({rate_per_min:.2f}/min, {total_below:.1f}s below threshold)")
        return result
        

def main():
    pass

if __name__ == '__main__':
    main()