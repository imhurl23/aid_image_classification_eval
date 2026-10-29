# Data and third-party material

The [MIT license](LICENSE) covers the original code and accompanying documentation
in this repository. It does not grant rights to third-party datasets, photographs,
papers, model weights, or provider responses. No source image binaries, downloaded
papers, or model checkpoints are included in the release.

- **AID (Aerial Image Dataset)**: the [authors' project page](https://captain-whu.github.io/AID/)
  describes 10,000 images in 30 classes, collected from Google Earth imagery.
  See Xia et al., *AID: A Benchmark Dataset for Performance Evaluation of Aerial
  Scene Classification*, IEEE TGRS 55(7), 2017, pp. 3965–3981. The code uses
  `blanchon/AID` revision `d9b5531544523f29d365a77a007b2ace3cea6882` and the
  `isaaccorley/aid` community split revision
  `e6767964e40f567a18eac6e6ca5fce397e4ce411`. A dataset redistribution license
  has not been established for this release; download source images separately
  and consult the original source terms before using or redistributing them.
- **LADI**: see the [MITLL dataset card](https://huggingface.co/datasets/MITLL/LADI-v2-dataset)
  and the pinned source revision in `outputs/ladi_eval/config.json` for attribution
  and applicable dataset/model terms.
- **Wikimedia Commons photos**: each file has its own attribution and license.
  Source pages and license metadata are retained in `outputs/aid_eval/web-aid-v1/`.
  Review the source file page before reusing a photo; retrieval categories are
  not human reference labels.
- **AiTLAS**: the [Benchmark Arena paper](https://arxiv.org/abs/2207.07189) and
  [source repository](https://github.com/biasvariancelabs/aitlas-arena) describe
  the specialist baseline. Checkpoint provenance and split manifests are retained
  under `outputs/aid_eval/specialist/aitlas/`; weights are downloaded separately.
- **Provider outputs**: the JSON/JSONL exports contain model predictions and API
  response data. Provider names identify the services used; they do not imply
  endorsement. Provider terms apply to access and use of their services.

Public links to historical Braintrust experiments may require access to the
original workspace. The local exports can be inspected without that access.
