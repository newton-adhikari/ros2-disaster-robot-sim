import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import yaml


def main():
    parser = argparse.ArgumentParser(description='Measure map coverage')
    parser.add_argument('pgm', help='Path to .pgm map file')
    parser.add_argument('--building-area', type=float, default=None,
                        help='Building interior area in m² (default: auto from map)')
    args = parser.parse_args()

    pgm = Path(args.pgm)
    yaml_path = pgm.with_suffix('.yaml')

    # Read resolution from YAML if available
    if yaml_path.exists():
        resolution = yaml.safe_load(open(yaml_path))['resolution']
    else:
        resolution = 0.05
        print(f'WARNING: No YAML found, assuming resolution={resolution} m/px')

    img = np.array(Image.open(pgm).convert('L'))
    cell = resolution ** 2   # m² per pixel

    free     = np.sum(img > 220)   # white = free
    occupied = np.sum(img < 50)    # black = obstacle
    unknown  = np.sum((img >= 50) & (img <= 220))

    free_m2     = free     * cell
    occupied_m2 = occupied * cell
    unknown_m2  = unknown  * cell
    known_m2    = free_m2 + occupied_m2

    # Building area: use argument, or 400m² design spec, but cap coverage at 100%
    DESIGN_AREA = 400.0   # 20m × 20m building interior
    if args.building_area:
        building_area = args.building_area
    else:
        # Use design spec — if robot stayed inside building
        # If free > design area, robot left the building (bad trial)
        building_area = DESIGN_AREA
        if free_m2 > DESIGN_AREA * 1.05:
            print(f'WARNING: Free area ({free_m2:.1f} m²) exceeds building design '
                  f'({DESIGN_AREA} m²). Robot may have left the building.')
            print(f'         Check if simulation was restarted before this trial.')
            print(f'         Using known map area as denominator instead.')
            building_area = known_m2

    coverage = min((free_m2 / building_area) * 100, 100.0)  # cap at 100%

    print(f'Map size:      {img.shape[1]} x {img.shape[0]} px @ {resolution}m/px')
    print(f'Free space:    {free_m2:.1f} m²')
    print(f'Obstacles:     {occupied_m2:.1f} m²')
    print(f'Unknown:       {unknown_m2:.1f} m²')
    print(f'Known area:    {known_m2:.1f} m²')
    print(f'Building area: {building_area:.1f} m²  (design spec: {DESIGN_AREA} m²)')
    print(f'COVERAGE:      {coverage:.1f}%')

    # Save coloured output
    out = np.stack([img, img, img], axis=2).astype(np.uint8)
    out[img > 220] = [144, 238, 144]   # green = explored free
    out[img < 50]  = [100,   0,   0]   # dark red = obstacle
    out_path = pgm.with_name(pgm.stem + '_coverage.png')
    Image.fromarray(out).save(out_path)
    print(f'Saved:  {out_path}')

    return coverage


if __name__ == '__main__':
    main()