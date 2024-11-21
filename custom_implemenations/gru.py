import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

class CustomGRU(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size):
        super(CustomGRU, self).__init__()

        # Taille de l'état caché
        self.hidden_size = hidden_size

        # Transforme l'entrée en embedding
        self.i2e = nn.Linear(input_size, emb_size)
        
        # Paramètres pour les portes GRU (Update, Reset) et la new state

        # Update Gate
        self.Wz = nn.Parameter(torch.Tensor(emb_size, hidden_size))  # Poids pour l'entrée (emb -> gru)
        self.Uz = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # Poids pour l'état caché (hidden -> gru)
        self.bz = nn.Parameter(torch.Tensor(hidden_size))  # Biais

        # Reset Gate
        self.Wr = nn.Parameter(torch.Tensor(emb_size, hidden_size))  # Poids pour l'entrée (emb -> gru)
        self.Ur = nn.Parameter(torch.Tensor(hidden_size, hidden_size))  # Poids pour l'état caché (hidden -> gru)
        self.br = nn.Parameter(torch.Tensor(hidden_size))  # Biais

        # New State
        self.W = nn.Parameter(torch.Tensor(emb_size, hidden_size))   # Poids pour l'entrée (emb -> gru)
        self.U = nn.Parameter(torch.Tensor(hidden_size, hidden_size)) # Poids pour l'état caché (hidden -> gru)
        self.b = nn.Parameter(torch.Tensor(hidden_size)) # Biais

        # Transforme l'état caché final en une prédiction
        self.fc = nn.Linear(hidden_size, output_size)

        # Appliquer la fonction log-softmax pour obtenir les probabilités des classes
        self.softmax = nn.LogSoftmax(dim=1)

        # Initialisation des poids
        self.init_weights()

    def init_weights(self):
        # Initialisation uniforme des poids avec une plage dépendante de la taille de l'état caché
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        # Obtenir la longueur de la séquence, la taille du batch et la taille du vocab
        seq_len, bs, _ = x.size()
        # Initialisation de l'état caché
        h_t = torch.zeros(bs, self.hidden_size)
        # Pour chaque mot de la séquence
        for t in range(seq_len):
            # Sélectionner le mot
            x_t = x[t]

            # Calcul de l'embedding
            e_t = self.i2e(x_t)

            # Calcul des portes
            # Update Gate -> Dans quelle mesure le nouvel état remplace l'ancien
            z_t = torch.sigmoid(e_t @ self.Wz + h_t @ self.Uz + self.bz)
            # Reset Gate -> Dans quelle mesure l'état caché précédent est utilisé
            r_t = torch.sigmoid(e_t @ self.Wr + h_t @ self.Ur + self.br)
            # New State -> Calcul du nouvel état temporaire
            h_hat_t = torch.tanh(e_t @ self.W + (r_t * h_t) @ self.U + self.b)

            # Mise à jour de l'état caché (combinaison de l'état précédent et du nouvel état temporaire)
            h_t = (1 - z_t) * h_t + z_t * h_hat_t 

        # La sortie finale est calculée à partir de l'état caché du dernier pas de temps
        output = self.fc(h_t)
        output = self.softmax(output)

        # Renvoyer les probabilités de classe
        return output

    def fit(self, train_loader, val_loader, epochs, lr, loss_func=nn.NLLLoss(), patience=3):
        # Intialisation de l'optimiseur
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        # Variables utilisées pour l'early stopping afin d'éviter l'overfitting
        previous_loss = float('-inf')
        epochs_with_no_improvement = 0
        # Dictionnaire pour garder les statistiques au cours de l'entrainement
        history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        # Pour chaque epoch
        for epoch in range(epochs):
            # Phase d'entrainement

            # Initialiser les variables pour les statistiques
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            # Passer le modèle en mode training
            self.train()

            # Pour chaque batch du loader
            for x, t in train_loader:
                # Reset du gradient
                optimizer.zero_grad()

                # Feed forward au modèle
                output = self(x)

                # Calcul de la loss
                loss = loss_func(output, t)

                # Descente de gradient et mise à jour des poids
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                # Incrémenter les statistiques
                train_correct += (torch.argmax(output, 1) == t).sum().item()
                train_total += t.size(0)

            # Calcul de la loss en prenant la moyenne des loss et calcul de l'accuracy
            avg_train_loss = train_loss / len(train_loader)
            avg_train_acc = train_correct / train_total

            # Phase de validation

            # Passer le modèle en mode evaluation
            self.eval()

            # Initialiser les variables pour les statistiques
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            # Désactiver le calcul de gradient
            with torch.no_grad():

                # Pour chaque batch du loader
                for x, t in val_loader:

                    # Feed forward au modèle
                    output = self(x)

                    # Calcul de la loss
                    loss = loss_func(output, t)

                    # Incrémenter les statistiques
                    val_loss += loss.item()
                    val_correct += (torch.argmax(output, 1) == t).sum().item()
                    val_total += t.size(0)

            # Calcul de la loss en prenant la moyenne des loss et calcul de l'accuracy
            avg_val_loss = val_loss / len(val_loader)
            avg_val_acc = val_correct / val_total

            # Mettre à jour le dictionnaire
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(avg_train_acc)
            history['val_acc'].append(avg_val_acc)

            # Afficher les résultats obtenus
            print(f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
            
            # Early stopping lorsqu'on ne remarque pas d'amélioration de la val_loss au cours de patience (default 3) epochs 
            if previous_loss <= avg_val_loss:
                epochs_with_no_improvement += 1
            else:
                epochs_with_no_improvement = 0
            previous_loss = avg_val_loss
            if epochs_with_no_improvement == patience:
                break

        # Renvoyer les statistiques d'entrainement
        return history

    def evaluate(self, test_loader, encoder, loss_func=nn.NLLLoss()):
        # Variables pour les statistiques
        correct = []
        actual_labels = [] 
        total_loss = 0

        # Passer le modèle en mode evaluation
        self.eval()

        # Désactiver le calcul de gradient
        with torch.no_grad():
            # Pour chaque batch du loader
            for x, t in test_loader:
                # Feed forward au modèle
                output = self(x)
                # Calcul de la loss
                loss = loss_func(output, t)
                # Incrémenter les statistiques
                total_loss += loss.item()
                correct.extend(torch.argmax(output, 1).cpu().numpy())
                actual_labels.extend(t.cpu().numpy())
            # Calcule de la loss moyenne
            avg_loss = total_loss / len(test_loader)
        # Afficher les statistiques
        print(f"Accuracy Score: {accuracy_score(actual_labels, correct)} - Loss: {avg_loss}")
        # On décode les labels
        actual_labels = encoder.inverse_transform(actual_labels)
        correct = encoder.inverse_transform(correct)
        # Afficher un classification report et une matrice de confusion
        print(f"Classification report\n {classification_report(actual_labels, correct)}")
        ConfusionMatrixDisplay.from_predictions(actual_labels, correct)
        plt.show()

    def save(self, path):
        """
            Cette fonction nous permet d'enregistrer le modèle
        """
        path = os.path.join('models', path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        """
            Cette fonction nous permet de charger les poids du modèle d'un fichier
        """
        if not os.path.exists(path):
            path = os.path.join('models', path)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Model file not found at {path}")
        self.load_state_dict(torch.load(path))