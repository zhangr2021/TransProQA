# LitTransProQA

LitTransProQA is an LLM-based QA metric for literary translation evaluation. 

## News:
 - 08.05.2025: First release of important code and copyright-free dataset (copyrighted data, complete codes, and reproducible pipeline with full data will be updated shortly).

## Project Structure

```
LitTransProQA/
├── datasets/                    # Dataset directory
│   ├── finetuning_dataset/     # Datasets for model fine-tuning
│   └── evaluation_set/         # Datasets for evaluation
├── finetuneing_method/         # Fine-tuning related code
│   ├── configs/               # Configuration files
│   ├── xcomet_regression.py   # Regression task
│   ├── xcomet_inference.py    # Inference implementation
│   └── xcomet_ranking.py      # Ranking task
├── prompting_method/          # Prompt-based approaches
│   ├── template/             # Prompt templates
│   ├── QA_translators/       # translator votes
│   ├── prompt_openrouter.py  # API integration
│   ├── run_all_models.py     # Model execution script
│   └── build_dataset.py      # Prompt preparation
└── SOTA_metric/              # State-of-the-art metrics
    └── m_prometheous.py      # Prometheus metric implementation
```

## Features

- **Multiple Assessment Methods**:
  - Fine-tuning based approaches using XCOMET
  - Prompt-based LitTransProQA: question-answering based translation evaluation
  - Other SOTA metrics

## Getting Started

### LiTranProQA Overview
![LitTransproQA summary](Fig/figure1.png)

### Prerequisites

- Python 3.8+
- Required packages (to be added to requirements.txt)

### Usage
```bash
To be updated
```

## License

This project is licensed under the MIT License.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{zhang2025litransproqallmbasedliterarytranslation,
      title={LiTransProQA: an LLM-based Literary Translation evaluation metric with Professional Question Answering}, 
      author={Ran Zhang and Wei Zhao and Lieve Macken and Steffen Eger},
      year={2025},
      eprint={2505.05423},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.05423}, 
}
```
