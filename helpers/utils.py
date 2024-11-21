import torch
import numpy as np
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

def load_file(file_path, sep):
    """
        Cette fonction permet de charger un fichier texte contenant le jeu de données de classification des sentiments sous format (<texte><sep><sentiment>) par ligne
    """
    # Listes pour stocker les textes et sentiments
    texts = []
    emotions = []

    # Lecture du fichier afin de récupérer toutes les lignes
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Répérer pour chacune des lignes
    for line in lines:
        # Récupérer le texte et l'émotion en séparant la ligne par le séparateur spécifié
        text, emotion = line.strip().split(sep)
        # Ajouter le texte et l'émotion ainsi obtenus à leur liste respectives
        texts.append(text)
        emotions.append(emotion)
    # Renvoyer la liste des textes et des émotions
    return texts, emotions

def create_vocab(sentences):
    """
        Cette fonction permet de créer un vocabulaire à partir de la liste des textes.          """
    def yield_lines():
        """
            Cette fonction nous permet de créer un itérateur sur nos phrases qui sera utilisé pour la création du vocabulaire
        """
        # Répéter pour chacune des phrases
        for sentence in sentences:
            # Renvoyer la liste des mots de la phrase (tokens)
            yield sentence.split()
    # Le vocabulaire est créé à partir de l'itérateur généré par la fonction interne yield_lines, en ajoutant le token spécial <unk>
    vocab = build_vocab_from_iterator(yield_lines(), specials=["<unk>"])
    # Par défaut utiliser l'indice du token <unk> pour les mots inconnus
    vocab.set_default_index(vocab['<unk>'])
    return vocab

def get_one_hot(vocab, keys, max_length=24):
    """
        Cette fonction nous permet de calculer une représentation de nos phrases sous forme de tenseur où chaque mot de la phrase est one hot encodé.
        On rembourre ou tronque également le vecteur suivant la taille de la phrase.
    """
    # Comme le vocabulaire attend une liste de mots, si un seul mot est fourni, il est transformé en une liste.    
    if isinstance(keys, str):
        keys = [keys]

    # Récupérer ensuite les indices des mots passées en entrée à partir du vocabulaire
    indices = vocab(keys)

    # Si le nombre d'indices obtenu est supérieur à la taille maximale défini alors il faut tronquer
    if len(indices) > max_length:
        indices = indices[:max_length]
    # Si le nombre d'indices obtenu est inférieur à la taille maximale définie, il faut rembourrer en ajoutant des zéros à la fin de la liste des indices.
    if len(indices) < max_length:
        padding = [0] * (max_length - len(indices))
        indices = indices + padding

    # Transformer la liste des indices ainsi obtenue en tenseur
    indices_tensor = torch.tensor(indices, dtype=torch.long)

    # Récupérer la représentation one-hot du tenseur en utilisant la fonction one_hot de torch.
    # Elle transforme chaque indice en un vecteur one-hot de taille vocab_size, et renvoie ainsi un tenseur de taille (max_length, vocab_size)
    one_hot_tensor = one_hot(indices_tensor, num_classes=len(vocab))

    # Renvoyer la représentation one hot de notre phrase
    return one_hot_tensor


class TextDataset(Dataset):
    """
        Cette classe nous permet de créer un jeu de données.
        Elle est utilisée pour redéfinir la méthode __getitem__, afin de retourner un texte et son label correspondant.
    """
    def __init__(self, texts, labels):
        # Initialisation avec les textes et emotions
        self.texts = texts
        self.labels = labels

    def __len__(self):
        # Redéfinir la taille du jeu de données
        return len(self.texts)

    def __getitem__(self, idx):
        # Redéfinir la méthode afin de renvoyer pour l'index donné la paire (texte,emotion) correspondante
        return self.texts[idx], torch.tensor(self.labels[idx])

def collate_fn(batch, vocab, max_length = 24):
    """
        Cette fonction nous permet d'effectuer une transformation sur nos batchs afin de les adapter à notre réseau
    """
    # Récupérer les textes et labels contenus dans le batch
    texts, labels = zip(*batch)
    # Pour chaque texte du batch générer son encodage one-hot et regrouper les encodages de chaque phrase ce qui nous donne notre tenseur (batch_size,max_length,vocab_size)
    one_hot_batch = torch.stack([get_one_hot(vocab, text.split(), max_length) for text in texts]).to(torch.float32)
    # Permutter les dimensions 0 et 1 afin d'obtenir le tenseur des textes sous forme (max_length,batch_size,vocab_size)
    one_hot_batch = one_hot_batch.permute(dims=(1,0,2))
    # Créer le tenseur des labels
    labels_batch = torch.from_numpy(np.array(labels)).to(torch.long)
    # Renvoyer le batch textes et labels ainsi obtenu
    return one_hot_batch, labels_batch


def create_loader(sentences, labels, vocab, batch_size, shuffle=False, max_length=24):
    """
        Cette fonction nous permet de créer un data loader afin de générer les batchs
    """
    # Création du jeu de données
    dataset = TextDataset(sentences, labels)
    # Création du loader
    # Nous utilisons la fonction précédemment créée afin d'appliquer les transformations nécessaires sur nos batchs lors du chargement
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=lambda batch: collate_fn(batch, vocab, max_length)
    )
    return loader

def calculate_mean_length(texts):
    """
        Cette fonction nous permet de calculer la taille moyenne des phrases dans une liste de phrases passée en entrée
    """
    # Calculer la somme des tailles de phrases
    total_length = sum(len(sentence.split()) for sentence in texts)
    # Calculer la moyenne en divisant cette somme par le nombre de phrases
    mean_length = total_length / len(texts) if texts else 0
    # Renvoyer la taille moyenne ainsi calculée
    return mean_length

def plot_train_statistics(history):
    """
        Cette fonction nous permet de créer un graphique permettant de montrer les résultats obtenus au cours de l'entrainement
    """
    # On récupère les résultats en terme de loss et d'accuracy sur la train et la validation en accèdant au dictionnaire
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    train_accuracy = history['train_acc']
    val_accuracy = history['val_acc']
    # Nombre d'epochs pour le graphique
    epochs = range(1, len(train_loss) + 1)

    # On créer deux subplots pour afficher les graphiques côte à côte
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Graphique de l'évolution de la loss au cours du training et du validation
    axes[0].plot(epochs, train_loss, label='Training Loss', color='blue')
    axes[0].plot(epochs, val_loss, label='Validation Loss', color='orange')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid()

    # Graphique de l'évolution de l'accuracy au cours du training et du validation
    axes[1].plot(epochs, train_accuracy, label='Training Accuracy', color='green')
    axes[1].plot(epochs, val_accuracy, label='Validation Accuracy', color='red')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid()

    # Affichage des graphiques
    plt.tight_layout()
    plt.show()