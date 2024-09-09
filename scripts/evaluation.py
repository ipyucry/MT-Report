import os
import sacrebleu
import nltk
from nltk.translate.meteor_score import meteor_score
import bert_score
import pandas as pd

# Ensure necessary NLTK resources are available
nltk.download('punkt')
nltk.download('wordnet')

def evaluate_bleu(reference, hypothesis):
    return sacrebleu.raw_corpus_bleu([hypothesis], [[reference]]).score

def evaluate_meteor(reference, hypothesis):
    # Tokenize the reference and hypothesis
    reference_tokens = nltk.word_tokenize(reference)
    hypothesis_tokens = nltk.word_tokenize(hypothesis)
    
    # Calculate METEOR score
    return meteor_score([reference_tokens], hypothesis_tokens)

def evaluate_bert_score(reference, hypothesis, lang='en'):
    P, R, F1 = bert_score.score([hypothesis], [reference], lang=lang)
    return P.mean().item(), R.mean().item(), F1.mean().item()

def evaluate_texts(base_dir, lang_pair, output_csv):
    results = []

    # List of translation services and their corresponding file names
    services = {
        'Google Translate': 'gt_translation.txt',
        'DeepL': 'dl_translation.txt',
        'Yandex Translate': 'yt_translation.txt',
        'Microsoft Translator': 'mt_translation.txt',
        'Reverso': 'rv_translation.txt'
    }
    
    # Mapping of folder prefixes to domain names
    domain_mapping = {
        'agri': 'Agriculture',
        'legal': 'Law',
        'mark': 'Marketing',
        'med': 'Medicine',
        'news': 'News',
        'fiction': 'Fiction',
        'nonfic': 'Nonfiction',
        'soc': 'Social Media Post',
        'tech': 'Technology',
    }

    # Traverse through all subfolders in the specified directory
    for subdir, _, files in os.walk(base_dir):
        if 'original.txt' in files and 'ref_translation.txt' in files:
            print(f"Evaluating {subdir}...")

            # Extract the domain from the folder name and map it using the domain_mapping dictionary
            folder_name = os.path.basename(subdir)
            domain_prefix = folder_name.split('_')[0]
            domain = domain_mapping.get(domain_prefix, 'Unknown')  # Default to 'Unknown' if not found

            # Check for 'long_' prefix and adjust domain accordingly
            if domain_prefix == 'long':
                actual_prefix = folder_name.split('_')[1]  # e.g., 'news' in 'long_news_1'
                domain = domain_mapping.get(actual_prefix, 'Unknown') + ' (long-form)'
            else:
                domain = domain_mapping.get(domain_prefix, 'Unknown')

            # Read the reference translation
            with open(os.path.join(subdir, 'ref_translation.txt'), 'r', encoding='utf-8') as ref_file:
                reference = ref_file.read().strip()

            # Evaluate each service
            for service_name, file_name in services.items():
                if file_name in files:
                    with open(os.path.join(subdir, file_name), 'r', encoding='utf-8') as hyp_file:
                        hypothesis = hyp_file.read().strip()

                    # Calculate BLEU, METEOR, and BERTScore
                    bleu_score = evaluate_bleu(reference, hypothesis)
                    meteor = evaluate_meteor(reference, hypothesis)
                    precision, recall, f1 = evaluate_bert_score(reference, hypothesis, lang=lang_pair[:2])

                    results.append({
                        'Text': folder_name,
                        'Domain': domain,  # Use the mapped domain here
                        'Service': service_name,
                        'BLEU': bleu_score,
                        'METEOR': meteor,
                        'BERT-Precision': precision,
                        'BERT-Recall': recall,
                        'BERT-F1': f1
                    })

    # Save results to a CSV file in the "results" directory
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Define the directories and language pairs relative to the script's location
base_path = os.path.dirname(os.path.dirname(__file__))  # Go up one directory level from the "scripts" folder
directories = [
    (os.path.join(base_path, 'text_samples/en-ru'), 'en-ru', os.path.join(base_path, 'results', 'results_en_ru.csv')),
    (os.path.join(base_path, 'text_samples/ru-en'), 'ru-en', os.path.join(base_path, 'results', 'results_ru_en.csv')),
    (os.path.join(base_path, 'text_samples/no-en'), 'no-en', os.path.join(base_path, 'results', 'results_no_en.csv'))
]

# Run evaluation for each directory
for base_dir, lang_pair, output_csv in directories:
    evaluate_texts(base_dir, lang_pair, output_csv)