# Korean lexical complexity analyzer

A python package for Korean lexical complexity analyzer.

## Installation

Install via pip:

```bash
pip install klca
```
	
## Usage

Analyze one file:

```bash
python3 -m klca file --input-file path/to/text.txt --output output.json
```

Analyze a folder:

```bash
python3 -m klca folder --input-dir path/to/texts --output results.csv
```

- Use `--recursive` to include text files in subfolders. Without it, only files directly inside `--input-dir` are processed.

## Included Resources
This package built on two open-source resources:
- Reference databases (korean-fineweb-edu) for calculating rarity, range, and bigram strength of association
- Vocabulary grade database (sourced from National Institute of Korean Language), released under Korea Open Government License Type 1

## Morpheme Parsing and Tagging
- By default, `klca` uses the Korean `stanza` GSD model for tokenization, POS tagging, and lemmatization.
- If you want to use a different Korean `stanza` model or a custom local model, you can modify the pipeline in the setting.

## Index Description
Detailed descriptions of the indices are available in the following [doc](./doc/Index_description_040926.pdf).

## Quick Demo
This is a quick [demo](https://huggingface.co/spaces/hksung/KLC-demo) - wake it up if it’s asleep (i.e., Click `Restart this Space`)

## Note
This package is currently in beta testing. Results may slightly change as the indices and resources continue to be refined. Please use with caution.

## License
This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
