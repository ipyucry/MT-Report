## Translation Evaluation Project

This project evaluates the performance of a number of translation services across multiple language pairs and domains as part of a written report for an NLP class. Specifically, we are evaluating RU-EN, EN-RU, and NO-EN translations using BLEU, METEOR, and BERTScore metrics.

## Project Structure
- `text_samples/`: Contains original texts, reference translations, and translations from various services. Texts are sorted by domains, with domain indicated in the folder name (i.e. med_1).
- `scripts/`: Python scripts for evaluating translation quality.
- `results/`: Stores output from evaluations.

## Dependencies
- Python 3.8+
- sacrebleu
- nltk
- textblob
- seaborn
- bert-score
- matplotlib
- pandas

## License
This project is licensed under the MIT License.