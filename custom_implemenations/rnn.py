import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
import matplotlib.pyplot as plt
import math
import warnings

warnings.filterwarnings('ignore')

class CustomRNN(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, output_size):
        super(CustomRNN, self).__init__()
        # Taille de l'etat caché
        self.hidden_size = hidden_size
        
        # Couches linéaires pour le calcul des etats cachés et des sorties

        # Transforme l'entrée en vecteur de taille emb_size
        self.i2e = nn.Linear(input_size, emb_size)

        # Transforme l'entrée et l'etat caché en un nouvel état caché
        self.i2h = nn.Linear(emb_size + hidden_size, hidden_size)

        # Transforme l'entrée et l'etat caché en sortie
        self.i2o = nn.Linear(emb_size + hidden_size, output_size)

        # Applique la fonction de log-softmax pour obtenir des probabilités de classe
        self.softmax = nn.LogSoftmax(dim=1)

        # Initialisation des poids
        self.init_weights()

    def init_weights(self):
        # Initialisation uniforme des poids avec une plage dépendante de la taille de l'état caché
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, hidden):
        # Calculer l'embedding
        emb = self.i2e(input)
        # Combiner l'entrée actuelle et l'etat caché
        combined = torch.cat((emb, hidden), 1)
        # Calculer l'état caché actuel avec une activation ReLU 
        hidden = F.relu(self.i2h(combined))
        # Calculer la sortie
        output = self.i2o(combined)
        # Appliquer la softmax à la sortie
        output = self.softmax(output)
        # Retourner la sortie et le nouvel etat caché
        return output, hidden

    def initHidden(self, batch_size):
        # Initialise l'état caché avec des zéros pour la taille de batch
        return Variable(torch.zeros(batch_size, self.hidden_size))

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
                # Initialiser l'état caché pour le batch
                hidden = self.initHidden(x.size(1))
                # Pour chaque mot de la sequence
                for i in range(x.size(0)):
                    # Calcul de la sortie et de l'etat caché
                    output, hidden = self(x[i], hidden)
                # Calcul de la loss en utilisant la dernière sortie
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
                    # Initialiser l'etat caché pour le batch
                    hidden = self.initHidden(x.size(1))
                    # Pour chaque mot de la sequence
                    for i in range(x.size(0)):
                        # Calcul de la sortie et de l'etat caché
                        output, hidden = self(x[i], hidden)
                    # Calcul de la loss en utilisant la dernière sortie
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
                # Initialiser l'etat caché pour le batch
                hidden = self.initHidden(x.size(1))
                # Pour chaque mot de la sequence
                for i in range(x.size(0)):
                    # Calcul de la sortie et de l'etat caché
                    output, hidden = self(x[i], hidden)
                # Calcul de la loss en utilisant la dernière sortie
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