"""Export the same authored SVG timeline to a portable GIF."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import subprocess
import tempfile

from PIL import Image

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('realvr_story', HERE / 'make_slide8_animation.py')
story = importlib.util.module_from_spec(spec)
spec.loader.exec_module(story)


def export(duration=story.DURATION, width=900, fps=10, output=None):
    total = round(duration * fps)
    output = Path(output) if output else HERE.parent / 'evidence-credit.gif'
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='realvr-story-') as directory:
        folder = Path(directory)

        def rasterize(index, source):
            result = subprocess.run(
                ['rsvg-convert', '-w', str(width)], input=source.encode(),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
            )
            frame = Image.open(io.BytesIO(result.stdout)).convert('RGB')
            frame.quantize(colors=192, method=Image.Quantize.MEDIANCUT,
                           dither=Image.Dither.NONE).save(folder / f'{index:04d}.png')

        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = []
            for index in range(total):
                pending.append(pool.submit(rasterize, index, story.render_svg(index / fps)))
                if len(pending) >= 8:
                    pending.pop(0).result()
                if index and index % 90 == 0:
                    print(f'Rendered {index}/{total} frames', flush=True)
            for task in pending:
                task.result()
        frames = [Image.open(folder / f'{index:04d}.png') for index in range(total)]
        durations = [round((i + 1) * 100 / fps) * 10 - round(i * 100 / fps) * 10
                     for i in range(total)]
        frames[0].save(output, save_all=True, append_images=frames[1:],
                       duration=durations, loop=0, optimize=True, disposal=1)
        for frame in frames:
            frame.close()
    with Image.open(output) as result:
        milliseconds = 0
        for i in range(result.n_frames):
            result.seek(i)
            milliseconds += result.info.get('duration', 0)
        print(f'{output}: {result.n_frames} frames, {milliseconds / 1000:.1f}s, '
              f'{output.stat().st_size / 1e6:.2f} MB', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=story.DURATION)
    parser.add_argument('--width', type=int, default=900)
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--output', type=Path, default=None)
    options = parser.parse_args()
    export(options.duration, options.width, options.fps, options.output)
