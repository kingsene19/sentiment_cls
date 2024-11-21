import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import string
from torch import nn
from sklearn.utils.class_weight import compute_class_weight
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download("punkt_tab")
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def get_rare_words_with_tfidf(all_texts):
    """
        Cette fonction nous permet de caluler les scores tfidf moyens de nos mots
    """
    # Initialiser le vecteur TF-IDF
    vectorizer = TfidfVectorizer()
    # Apprendre le vocabulaire et transformer les textes en matrice TF-IDF
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    # Récupérer les noms des mots (les features)
    feature_names = vectorizer.get_feature_names_out()
    # Calculer les scores TF-IDF moyens pour chaque mot sur tous les documents
    tfidf_means = np.mean(tfidf_matrix.toarray(), axis=0)
    return feature_names, tfidf_means

def plot_tfidf_distribution(all_texts):
    """
    Cette fonction calcule les scores TF-IDF pour les mots dans la liste de textes et affiche le graphique de la distribution tfidf
    """
    # Récupérer les score tf-idf moyens de nos mots
    feature_names, tfidf_means = get_rare_words_with_tfidf(all_texts)

    # Afficher la courbe des scores TF-IDF triés
    plt.figure(figsize=(12,6))
    plt.plot(np.sort(tfidf_means), color='green')
    plt.title('Courbe des scores TF-IDF moyens')
    plt.xlabel('Nombre de mots')
    plt.ylabel('Score TF-IDF moyen')
    
    # Afficher les graphiques
    plt.tight_layout()
    plt.show()

    return feature_names, tfidf_means

def remove_rare_words(text, rare_words):
    """
        Cette fonction nous permet de supprimer les mots rares d'un texte donné
    """
    # Nous commençons par ne garder que les mots du texte qui ne sont pas dans notre liste des mots rare
    after_rare_words = [word.lower() for word in text.split() if word not in rare_words]
    # On reforme le texte avec les mots qui ne sont pas rares
    text_post_rare_words = " ".join(word for word in after_rare_words)
    # Retourner le texte ainsi obtenu
    return text_post_rare_words

def clean_text(texts, rare_words):
    """
        Cette fonction nous permet de nettoyer nos textes
    """
    # La lemmatisation nous permet de transformer les mots en leur forme de base ou de dictionnaire en utilisant le contexte
    lemmatizer = WordNetLemmatizer()
    # Liste pour stocker les textes nettoyés
    cleaned_texts = []
    for text in texts:
        # Suppression de la ponctuation
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(f"[{string.punctuation}]", "", text)
        # Suppression des mots rares
        text = remove_rare_words(text, rare_words)
        # Suppression des stop words
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in text.split() if word not in stop_words]
        # Lemmatisation
        lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
        # Reformer le texte
        text = " ".join(lemmatized_tokens)
        # Ajouter le texte à liste des textes nettoyés
        cleaned_texts.append(text)
    # Renvoyer les textes nettoyés
    return cleaned_texts

def retrieve_class_weights(labels, unique_labels):
    """
       Déterminer une estimation des poids de chaque afin de contrer l'effet du déséquilibre des classes
    """
    # Estimer les poids de classe pour notre jeu de données
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    # Conversion en tenseur pour pouvoir être utilisée dans notre loss
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    # Renvoyer les poids
    return class_weights

class WeightedNLLLoss(nn.Module):
    """
        Nous créons une classe pour notre loss adaptée afin de prendre en compte les poids pour chacune de nos classes
    """
    def __init__(self, class_weights):
        super(WeightedNLLLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, outputs, labels):
        ce_loss = nn.NLLLoss(weight=self.class_weights)(outputs, labels)
        return ce_loss