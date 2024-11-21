from helpers.utils import load_file, create_vocab, create_loader, calculate_mean_length, plot_train_statistics
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from custom_implemenations.rnn import CustomRNN

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

    # Création du vocabulaires
    vocab = create_vocab(train_texts)

    # Récupérer la taille moyenne des phrases et prendre l'arrondi supérieur comme argument pour max_length
    mean_length = calculate_mean_length(train_texts)
    max_length = math.ceil(mean_length)

    # Création des data loader
    batch_size = 96
    train_loader = create_loader(train_texts, train_emotions, vocab, batch_size=batch_size, shuffle=True, max_length=max_length)
    val_loader = create_loader(val_texts, val_emotions, vocab, batch_size=batch_size, max_length=max_length)
    test_loader = create_loader(test_texts, test_emotions, vocab, batch_size=1, max_length=max_length)
    
    # Entrainement du modèle
    rnn = CustomRNN(len(vocab), 512, 128, len(np.unique(train_emotions)))
    history = rnn.fit(train_loader, val_loader, 10, lr=0.01)

    # Affichage des statistiques d'entrainement
    plot_train_statistics(history)

    # Evaluation du modèle
    rnn.evaluate(test_loader, le)

    # Sauvegarde du modèle
    rnn.save("initial_rnn.pth")