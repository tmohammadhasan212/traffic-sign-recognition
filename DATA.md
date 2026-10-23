# Data provenance and split policy

The dataset is the [German Traffic Sign Recognition Benchmark public archive](https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html), archive created 2019-05-10. Its authors are Johannes Stallkamp, Marc Schlipsing, Jan Salmen, and Christian Igel. This project uses `GTSRB_Final_Training_Images.zip`, `GTSRB_Final_Test_Images.zip`, and `GTSRB_Final_Test_GT.zip`. The archives used for the local pipeline check were downloaded on **2026-10-23**. Their SHA-256 hashes, in that order, are `d32ac4b5fa9a1cbd1994768413902e8193599d9434cf0a8eb9cfd00a6d3a290c`, `48ba6fab7e877eb64eaf8de99035b0aaecfbc279bee23e35deca4ac1d0a837fa`, and `f94e5a7614d75845c74c04ddb26b8796b9e483f43541dd95dd5b726504e16d6d`. The preparation script records each download's actual time and SHA-256 in `data/raw/download_provenance.json`. This repository does not redistribute images, annotations, or pretrained weights. The archive page does not state an explicit redistribution license; check the source terms before distributing derivatives.

Run `python scripts/prepare_data.py --download`. Expected layout after extraction:

```text
data/raw/GTSRB/Final_Training/Images/00000/GT-00000.csv
data/raw/GTSRB/Final_Training/Images/00000/00000_00000.ppm
data/raw/GTSRB/Final_Test/Images/00000.ppm
data/raw/GT-final_test.csv
```

The script also accepts the test CSV inside `Final_Test/Images`. It checks all 43 class folders, semicolon CSV annotations, file existence and uniqueness, image decoding, class IDs, disk/CSV alignment, and the expected 39,209 train and 12,630 test images. The archive can be placed manually under `data/raw/`; omit `--download` then.

The training filename's first five digits identify a physical sign track *within its class*. The split key is `<class>/<track>`. We sort tracks per class, shuffle with NumPy `default_rng(2026)`, and reserve 20% of each class's tracks for validation. All images from a track stay together. Augmentation happens only in the training Dataset after this split. The test images remain in a separate manifest section, and no test labels are used for training, checkpoint choice, temperature, or threshold choice. `configs/split.json` commits the policy and `configs/gtsrb_track_split.json` commits the generated official track assignments after first preparation. Later preparation checks this file and rejects a different archive or split. `data/split_manifest.json` is the generated file-level manifest and is ignored because it contains the external dataset's file inventory. Keep it with experiment artifacts.

The complete class ID to English name mapping is in `src/traffic_signs/data/labels.py`. It follows the common GTSRB 0–42 order. No horizontal flip is used because it can change sign meaning. Images are RGB, padded to a square with black borders to preserve aspect ratio, then resized by Pillow's default `ImageOps.pad` resampling to 64 × 64 for the scratch CNN or 224 × 224 for ResNet-18. ResNet uses ImageNet channel mean and standard deviation as documented by [torchvision's ResNet18 weights](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html).

Smoke fixtures under `tests/` are synthetic and are not GTSRB observations. They test parsing and leakage behavior without shipping dataset images.
