import os
import time
from concurrent import futures

import yaml
from pdf2image import convert_from_path


def convert_pdf_to_jpg(pdf_path, output_dir=None, dpi=200, n_thread=2):
    try:
        images = convert_from_path(pdf_path, dpi=dpi,
                                   thread_count=n_thread,
                                   output_folder=None,
                                   fmt="jpg")
    except:
        print(f"Error: {pdf_path}")
        return None

    if output_dir is not None:
        filename = os.path.basename(pdf_path).replace(".pdf", "")
        for i, x in enumerate(images):
            x.save(os.path.join(output_dir, f"{filename}-{i}.jpg"))

    return images


def batch_convert_pdfs_to_jpgs(input_dir, output_dir, dpi=200, n_thread=1):
    t_start = time.time()
    tot = 0

    with futures.ProcessPoolExecutor() as pool:
        for dirpath, dirnames, filenames in os.walk(input_dir):
            for dirname in dirnames:
                os.makedirs(os.path.join(dirpath.replace(input_dir, output_dir), dirname), exist_ok=True)

            files = list(filter(lambda x: x.endswith(".pdf"), filenames))
            if len(files) == 0:
                continue

            out_folder = dirpath.replace(input_dir, output_dir)
            print(f"Converting {len(files)} pdfs to {out_folder}...")

            for f in files:
                pool.submit(convert_pdf_to_jpg(os.path.join(dirpath, f), out_folder, dpi=dpi, n_thread=n_thread))

            tot += len(files)

    print(f"Collected {tot} pdfs in {time.time() - t_start}s.")

if __name__ == '__main__':
    with open("../../config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    batch_convert_pdfs_to_jpgs(os.path.join(config["ROOT"], config["PDF"]["INPUT_DIR"]),
                               os.path.join(config["ROOT"], config["PDF"]["OUTPUT_DIR"]))
