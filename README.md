# Korean lexical complexity analyzer

A python package for Korean lexical complexity analyzer.

## Installation

Install via pip:

```bash
pip install klca
```
	
## Usage

Analyze one file (with `txt` extension):

```bash
python3 -m klca file --input-file path/to/text.txt --output output.json
```

Analyze multiple texts in a folder:

```bash
python3 -m klca folder --input-dir path/to/texts --output results.csv
```

- Use `--recursive` to include text files in subfolders. Without it, only files directly inside `--input-dir` are processed.

## Included resources
This package built on two open-source resources:
- Reference databases (korean-fineweb-edu) for calculating rarity, range, and bigram strength of association
- Vocabulary grade database (sourced from National Institute of Korean Language), released under Korea Open Government License Type 1

## Built-in NLP tool for preprocessing
- By default, `klca` uses the Korean `stanza` GSD model for tokenization, POS tagging, and lemmatization.
- If you want to use a different Korean `stanza` model or a custom local model, you can modify the pipeline in the setting.

## Index description
Detailed descriptions of the indices are available in the following [doc](./doc/Index_description.pdf).

## Quick Demo
This is a quick web [demo](https://huggingface.co/spaces/hksung/KLC-demo) - wake it up if it’s asleep (i.e., Click `Restart this Space`)

## Citation
For more information about the analyzer, please see the following paper. 
  * Sung, H., & Shin, G.-H. (2026). [Developing and Validating Lexical Complexity Indices for Korean](https://doi.org/10.1016/j.rmal.2026.100359). *Research Methods in Applied Linguistics*.
  
If you use the tool in your research, we would greatly appreciate it if you could cite the paper.

# License
- This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
