from helpers.utils import load_file, create_vocab, create_loader, calculate_mean_length, plot_train_statistics
from helpers.optimisation import retrieve_class_weights, WeightedNLLLoss, clean_text, plot_tfidf_distribution
import math
import numpy as np
from sklearn.preprocessing import LabelEncoder
from custom_implemenations.gru import CustomGRU
from itertools import product

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
    # Nous allons donc utiliser un seuil faible (0.0001) afin d'avoir un bon équilibre entre la réduction d'information initule et la taille du vocabulaire
    rare_words = [feature_names[i] for i in range(len(tfidf_means)) if tfidf_means[i] < 0.0001]
    print(f"Nombre de mots rare {len(rare_words)}")

    # Prétraitement du texte
    train_texts = clean_text(train_texts, rare_words)
    val_texts = clean_text(val_texts, rare_words)
    test_texts = clean_text(test_texts, rare_words)

    # Création du vocabulaire
    vocab = create_vocab(train_texts)

    # Récupérer la taille moyenne des phrases et prendre l'arrondi supérieur comme argument pour max_length
    mean_length = calculate_mean_length(train_texts)
    max_length = math.ceil(mean_length) + 5

    # Utilisation de notre loss adaptée afin de tenir compte du déséquilibre des classes
    loss_func = WeightedNLLLoss(class_weights=class_weights)

    # Hyperparamètres à tester
    batch_sizes = [32, 64, 96]
    embedding_sizes = [128, 256, 512]
    learning_rates = [0.001, 0.005, 0.01]
    hidden_sizes = [64, 128, 256]

    # Stockage des résultats
    best_val_acc = 0
    best_params = None
    best_model = None

    # Recherche d'hyperparamètres
    for batch_size, lr, hidden_size, embedding_size in product(batch_sizes, learning_rates, hidden_sizes, embedding_sizes):
        print(f"Test batch_size={batch_size}, learning_rate={lr}, hidden_size={hidden_size}, embedding_size={embedding_size}")
        
        # Création des data loader avec le batch_size actuel
        train_loader = create_loader(train_texts, train_emotions, vocab, batch_size=batch_size, shuffle=True, max_length=max_length)
        val_loader = create_loader(val_texts, val_emotions, vocab, batch_size=batch_size, max_length=max_length)
        
        # Création et entraînement du modèle
        gru = CustomGRU(len(vocab), embedding_size, hidden_size, len(np.unique(train_emotions)))

        # Entrainement sur 5 epochs pour déterminer le meilleur modèle sur le val_set
        history = gru.fit(train_loader, val_loader, 5, lr=lr, loss_func=loss_func)
        
        # Récupérer l'accuracy sur le jeu de validation
        val_acc = max(history['val_acc'])
        print(f"Validation accuracy: {val_acc}")
        
        # Vérifier si cette combinaison est la meilleure jusqu'à présent
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = (batch_size, lr, hidden_size, embedding_size)
            best_model = gru

    print(f"Best Params: batch_size={best_params[0]}, learning_rate={best_params[1]}, hidden_size={best_params[2]}, embedding_size={best_params[3]} with validation accuracy={best_val_acc}")

    # Réentraînement avec les meilleurs hyperparamètres trouvés sur 10 epochs de plus
    batch_size, lr, _, _ = best_params
    train_loader = create_loader(train_texts, train_emotions, vocab, batch_size=batch_size, shuffle=True, max_length=max_length)
    val_loader = create_loader(val_texts, val_emotions, vocab, batch_size=batch_size, max_length=max_length)
    history = best_model.fit(train_loader, val_loader, 10, lr=lr)

    # Afficher les statistiques d'entraînement
    test_loader = create_loader(test_texts, test_emotions, vocab, batch_size=1, max_length=max_length)
    plot_train_statistics(history)

    # Évaluation du modèle sur le jeu de test
    best_model.evaluate(test_loader, le)

    # Sauvegarder le modèle
    best_model.save("best_model.pth")