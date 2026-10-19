"""Raster checks for rectangular canvases; all output is temporary."""
import contextlib
import importlib
import io
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/data_generation'))
g = importlib.import_module('DataSetGeneration')


class GeometryTests(unittest.TestCase):
    def assert_box_matches(self, image, data):
        yy, xx = np.where(np.any(image != 180, axis=2))
        self.assertGreater(xx.size, 0)
        expected = [xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]
        actual = [data['x'] - data['w'] / 2, data['y'] - data['h'] / 2,
                  data['x'] + data['w'] / 2, data['y'] + data['h'] / 2]
        np.testing.assert_allclose(actual, expected, atol=1)
        self.assertGreaterEqual(actual[0], 0)
        self.assertGreaterEqual(actual[1], 0)
        self.assertLessEqual(actual[2], image.shape[1])
        self.assertLessEqual(actual[3], image.shape[0])

    def test_rotated_polygons_match_pixels_on_all_aspect_ratios(self):
        for height, width in [(320, 320), (320, 640), (640, 320)]:
            for shape in ['Hexagon', 'Triangle']:
                for angle in range(0, 360, 15):
                    with self.subTest(canvas=(width, height), shape=shape, angle=angle):
                        image = np.full((height, width, 3), 180, np.uint8)
                        dimension = g.calculate_poly_dimension(image.shape, shape, .3, angle)
                        translation = [(width - dimension[0]) / 2, (height - dimension[1]) / 2]
                        result, data, _ = g.draw_shape(image, shape, 'A', .3, 1, translation,
                                                     angle, 0, (0, 255, 0), (0, 0, 0), 1, 0, False)
                        self.assertIsNotNone(result)
                        self.assert_box_matches(result, data)

    def test_polygon_border_included_in_box(self):
        for height, width in [(320, 640), (640, 320)]:
            image = np.full((height, width, 3), 180, np.uint8)
            result, data, _ = g.draw_shape(image, 'Hexagon', 'A', .3, 1,
                                          (50, 50), 45, 0, (0, 255, 0),
                                          (0, 0, 0), 1, 5, False)
            self.assertIsNotNone(result)
            self.assert_box_matches(result, data)

    def test_clipped_polygon_is_rejected(self):
        result = g.draw_shape(np.full((100, 200, 3), 180, np.uint8),
                              'Hexagon', 'A', .5, 1, [190, 90], 0, 0,
                              (0, 255, 0), (0, 0, 0), 1, 0, False)
        self.assertEqual(result, (None, None, None))

    def test_letter_pixels_and_box_on_rectangles(self):
        old = dict(g.sh.Letter_Mask)
        try:
            mask = np.zeros((100, 100), np.uint8)
            mask[20:80, 30:70] = 255
            g.sh.Letter_Mask['fixture'] = mask
            for height, width in [(320, 640), (640, 320)]:
                image = np.full((height, width, 3), 180, np.uint8)
                result, data = g.sh.add_letter(image, 'fixture', (0, 0, 0), .2,
                                              (width / 2, height / 2), 35)
                self.assert_box_matches(result, data)
        finally:
            g.sh.Letter_Mask.clear()
            g.sh.Letter_Mask.update(old)

    def test_circle_sectors_wrapped_rotation_bounds(self):
        for angles in [(0, 360), (330, 510), (350, 440)]:
            image = np.full((320, 640, 3), 180, np.uint8)
            result, data, _ = g.sh.Circle(image, angles, .3, (0, 255, 0), (100, 100), 0)
            self.assert_box_matches(result, data)

    def test_real_writer_normalizes_axes_and_fails_oversized_shapes(self):
        # The resize wrapper preserves fixture dimensions when normal generation
        # enforces a square SIZE; letter initialization is bypassed (no letters).
        with tempfile.TemporaryDirectory(prefix='foxtrot-geometry-') as directory:
            root = Path(directory)
            for height, width in [(320, 320), (320, 640), (640, 320)]:
                fixture = root / f'{width}x{height}'
                fixture.mkdir()
                cv2.imwrite(str(fixture / 'background.png'), np.full((height, width, 3), 180, np.uint8))
                values = {'Background': str(fixture), 'N': 10, 'ADD_LETTER': False,
                          'NOISE_FACTOR': 0, 'BLUR_FACTOR': [0, 0], 'ALT_BLUR_PROB': 0,
                          'SATURATION': [0, 0], 'SHAPE_SCALE': [.2, .4], 'DEBUG': True,
                          'LETTER_DATASET': False, 'ONLY_LETTER': False, 'DISPLAY': False,
                          'SAVE_BOUNDS': False, 'ADD_LIGHT_GRADIENT': False}
                for name in ['Output', 'Labels', 'Bounds', 'LetterOutput', 'LetterLabels', 'LetterBounds']:
                    path = fixture / name
                    path.mkdir()
                    values[name] = str(path)
                raw = []
                original = g.draw_shape
                def capture(*args, **kwargs):
                    result = original(*args, **kwargs)
                    if result[0] is not None:
                        self.assert_box_matches(result[0], result[1])
                        raw.append(result[1])
                    return result
                random.seed(42)
                np.random.seed(42)
                with patch.multiple(g, **values), patch.object(g, 'draw_shape', capture), \
                     patch.object(g.sh, 'initialize_letter_masks'), patch.object(g.os, 'system'), \
                     patch.object(g.cv2, 'resize', side_effect=lambda image, *a, **k: image), \
                     contextlib.redirect_stdout(io.StringIO()):
                    g.main()
                self.assertEqual(len(raw), 20)
                for index, data in enumerate(raw):
                    fields = (fixture / 'Labels' / f'{index}.txt').read_text().split()
                    np.testing.assert_allclose(list(map(float, fields[1:])),
                                               [data['x'] / width, data['y'] / height,
                                                data['w'] / width, data['h'] / height])
                values.update(N=1, SHAPE_SCALE=[5., 5.])
                with patch.multiple(g, **values), patch.object(g.sh, 'initialize_letter_masks'), \
                     patch.object(g.os, 'system'), contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(RuntimeError, 'Generated only'):
                        g.main()


if __name__ == '__main__':
    unittest.main()
