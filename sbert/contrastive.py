import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, util, evaluation

TRAIN_FILES = ["../data/ancient-greek-datasets/translation_pairs_a.txt", "../data/ancient-greek-datasets/translation_pairs_b.txt"]
EVAL_FILE = "../data/ancient-greek-datasets/translation_pairs_eval.txt"
OUTPUT_PATH = './output/ancient-greek-contrastive'

# Hyperparameters (Krahn et al. Table 9)
BATCH_SIZE = 82
EPOCHS = 10
WARMUP_STEPS = 2000
MAX_SEQ_LENGTH = 128
LEARNING_RATE = 2e-5

def load_data():
    examples = []
    for file_path in TRAIN_FILES:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                greek, english = line.strip().split('\t')
                examples.append(InputExample(texts=[greek, english]))
    return examples

def load_eval_data():   
    greek_sentences = []
    english_sentences = []
    with open(EVAL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            greek, english = line.strip().split('\t')
            greek_sentences.append(greek)
            english_sentences.append(english)
    return greek_sentences, english_sentences

def train():
    base_model = SentenceTransformer('../hf_format120')
    train_examples = load_data()
    train_dataloader = DataLoader(
        train_examples, 
        shuffle=True, 
        batch_size=BATCH_SIZE
    )

    eval_greek, eval_english = load_eval_data()
    evaluator = evaluation.TranslationEvaluator(
        source_sentences=eval_greek,
        target_sentences=eval_english,
        name='grc-en-eval',
        batch_size=BATCH_SIZE
    )

    train_loss = losses.MultipleNegativesRankingLoss(model=base_model)
    base_model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=EPOCHS,
        warmup_steps=WARMUP_STEPS,
        evaluator=evaluator,
        evaluation_steps=500,  # Evaluate every 500 steps
        output_path=OUTPUT_PATH,
        save_best_model=True,
        optimizer_params={'lr': LEARNING_RATE},
        show_progress_bar=True
    )

def eval(model):
    model.eval()
    greek_sentences, english_sentences = load_eval_data()
    greek_embeddings = model.encode(greek_sentences, convert_to_tensor=True, batch_size=BATCH_SIZE)
    english_embeddings = model.encode(english_sentences, convert_to_tensor=True, batch_size=BATCH_SIZE)
    cos_scores = util.cos_sim(greek_embeddings, english_embeddings)

    # greek to english
    correct = 0
    for i in range(len(greek_sentences)):
        best_match_idx = torch.argmax(cos_scores[i]).item()
        if best_match_idx == i: 
            correct += 1
    accuracy = correct / len(greek_sentences)
    print(f"Greek→English Accuracy: {accuracy:.4f}")

    # english to greek
    correct = 0
    for i in range(len(english_sentences)):
        best_match_idx = torch.argmax(cos_scores[:, i]).item()
        if best_match_idx == i:
            correct += 1
    accuracy_reverse = correct / len(english_sentences)
    print(f"English→Greek Accuracy: {accuracy_reverse:.4f}")

    # Average both directions (this is what they report in Table 2)
    avg_accuracy = (accuracy + accuracy_reverse) / 2
    print(f"Bidirectional Average: {avg_accuracy:.4f}")

def main():
    train()
    model = SentenceTransformer(OUTPUT_PATH)
    eval(model)

if __name__ == "__main__":
    main()