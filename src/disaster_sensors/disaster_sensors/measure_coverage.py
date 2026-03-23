import sys, numpy as np
from pathlib import Path
from PIL import Image
import yaml

pgm = Path(sys.argv[1])
yaml_path = pgm.with_suffix('.yaml')
resolution = yaml.safe_load(open(yaml_path))['resolution'] if yaml_path.exists() else 0.05

img = np.array(Image.open(pgm).convert('L'))
free     = np.sum(img > 220)
occupied = np.sum(img < 50)
unknown  = np.sum((img >= 50) & (img <= 220))
cell = resolution ** 2

free_m2 = free * cell
building_area = 400.0  # 20m x 20m building interior
coverage = (free_m2 / building_area) * 100

print(f"Map size:      {img.shape[1]} x {img.shape[0]} px @ {resolution}m/px")
print(f"Free space:    {free_m2:.1f} m²")
print(f"Obstacles:     {occupied * cell:.1f} m²")
print(f"Unknown:       {unknown * cell:.1f} m²")
print(f"COVERAGE:      {coverage:.1f}%")

# Save output
out = np.stack([img,img,img],axis=2).astype(np.uint8)
out[img > 220] = [144, 238, 144]   # green = explored free
out[img < 50]  = [100,   0,   0]   # dark red = obstacle
Image.fromarray(out).save(pgm.with_name(pgm.stem + '_coverage.png'))
print(f"Saved:  {pgm.with_name(pgm.stem + '_coverage.png')}")