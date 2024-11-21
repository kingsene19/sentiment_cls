from helpers.utils import load_file, plot_train_statistics
from helpers.optimisation import retrieve_class_weights, clean_text, plot_tfidf_distribution
import numpy as np
from sklearn.preprocessing import LabelEncoder
from helpers.bert_utils import WeightedCrossEntropyLoss, TextClassificationDataset, DistilBERTClassifier, train, evaluate
from transformers import DistilBertTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import torch


if __name__ == "__main__":

    # Récupérer les listes de paires (texte,emotion) pour chaque jeu de données
    train_texts, train_emotions = load_file('dataset/train.txt',';')
    val_texts, val_emotions = load_file('dataset/val.txt',';')
    test_texts, test_emotions = load_file('dataset/test.txt',';')

    # Encodage des labels
    le = LabelEncoder()
    train_emotions = le.fit_transform(train_emotions)
    val_emotions = le.transform(val_emotions)
    test_emotions = le.transform(test_emotions)

    # Estimer les poids des classes
    class_weights = retrieve_class_weights(train_emotions,np.unique(train_emotions))

    # Visualiser la distribution de fréquences des mots
    feature_names, tfidf_means = plot_tfidf_distribution(train_texts)
    # Après visualisation du graphique la majorité des mots apparaissent peu de fois, plus de 14000 mots avec une fréquence inférieure à 0.01 et très proches de 0
    # Nous allons donc utiliser un seuil faible (0.0005) afin d'avoir un bon équilibre entre la réduction d'information initule et la taille du vocabulaire
    rare_words = [feature_names[i] for i in range(len(tfidf_means)) if tfidf_means[i] < 0.0001]
    print(f"Nombre de mots rare {len(rare_words)}")
    # Prétraitement des textes
    train_texts = clean_text(train_texts, rare_words)
    val_texts = clean_text(val_texts, rare_words)
    test_texts = clean_text(test_texts, rare_words)

    # Utilisation de notre loss adaptée afin de tenir compte du déséquilibre des classes
    criterion = WeightedCrossEntropyLoss(class_weights)

    # Variable pour la préparation du modèle
    bert_model_name = 'distilbert/distilbert-base-uncased'
    num_classes = len(np.unique(train_emotions))
    max_length = 64
    batch_size = 96
    num_epochs = 3
    learning_rate = 2e-5

    # Initialisation du tokenizer et des loaders
    tokenizer = DistilBertTokenizer.from_pretrained(bert_model_name)
    train_dataset = TextClassificationDataset(train_texts, train_emotions, tokenizer, max_length)
    val_dataset = TextClassificationDataset(val_texts, val_emotions, tokenizer, max_length)
    test_dataset = TextClassificationDataset(test_texts, test_emotions, tokenizer, max_length)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialisation du modèle, de l'optimiseur et du scheduler
    model = DistilBERTClassifier(bert_model_name, num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    # Entrainement du modèle
    history = train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, num_epochs)

    # Affichage des statistiques d'entrainement
    plot_train_statistics(history)

    # Evaluation du modèle
    evaluate(model, test_dataloader, le, criterion)

    # Sauvegarde du modèle
    torch.save(model.state_dict(), "models/distilbert_classifier.pth")
    