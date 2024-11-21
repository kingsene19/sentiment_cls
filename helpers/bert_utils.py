import torch
from transformers import DistilBertForSequenceClassification
from torch.utils.data import Dataset
import torch
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class TextClassificationDataset(Dataset):
    """
        Cette classe nous permet de créer un jeu de données.
        Elle est utilisée pour redéfinir la méthode __getitem__, afim d'effectuer la tokenisation et retourner le bon format des données
    """
    def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
        return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label)}
    
class DistilBERTClassifier(torch.nn.Module):
    """
        Cette classe nous permet de crééer notre classifieur sur la base de notre DistilBert préentrainé 
    """
    def __init__(self, bert_model_name, num_classes):
        super(DistilBERTClassifier, self).__init__()
        self.bert = DistilBertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
    
class WeightedCrossEntropyLoss(torch.nn.Module):
    """
        Nous créons une classe pour notre loss adaptée afin de prendre en compte les poids pour chacune de nos classes
    """
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.class_weights = class_weights

    def forward(self, outputs, labels):
        ce_loss = torch.nn.CrossEntropyLoss(weight=self.class_weights)(outputs, labels)
        return ce_loss
    
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, patience=3):

    # Variables utilisées pour l'early stopping afin d'éviter l'overfitting
    previous_val_loss = float('inf')
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

        # Passer le modèle en mode training
        model.train()

        # Initialiser les variables pour les statistiques
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Pour chaque batch du loader
        for batch in train_loader:
            # Reset du gradient
            optimizer.zero_grad()
            # Récupérer les informations du batch
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            # Forward pass au modèle
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Calcul de la loss
            loss = criterion(outputs, labels)
            # Descente de gradient et mise à jour des poids
            loss.backward()
            optimizer.step()
            scheduler.step()
            # Incrémenter les statistiques
            train_loss += loss.item()
            train_correct += (torch.argmax(outputs, 1) == labels).sum().item()
            train_total += labels.size(0)
        
        # Calcul de la loss en prenant la moyenne des loss et calcul de l'accuracy
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / train_total

        # Phase de validation

        # Passer le model en mode evaluation
        model.eval()

        # Initialiser les variables pour les statistiques
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        # Désactiver le calcul de gradient
        with torch.no_grad():
            # Pour chaque batch du loader
            for batch in val_loader:
                # Récupérer les informations du batch
                input_ids = batch['input_ids']
                attention_mask = batch['attention_mask']
                labels = batch['label']
                # Forward pass au modèle
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Calcul de la loss
                loss = criterion(outputs, labels)
                # Incrémenter les statistiques
                val_loss += loss.item()
                val_correct += (torch.argmax(outputs, 1) == labels).sum().item()
                val_total += labels.size(0)

        # Calcul de la loss de l'epoch en prenant la moyenne des loss et calcul de l'accuracy
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total

        # Mettre à jour le dictionnaire
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(avg_train_acc)
        history['val_acc'].append(avg_val_acc)
        
        # Afficher les résultats obtenus
        print(f"Epoch {epoch + 1}/{epochs}, "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")


        # Early stopping lorsqu'on ne remarque pas d'amélioration de la val_loss au cours de patience (default 3) epochs 
        if previous_val_loss <= avg_val_loss:
            epochs_with_no_improvement += 1
        else:
            epochs_with_no_improvement = 0
        previous_val_loss = avg_val_loss
        if epochs_with_no_improvement == patience:
            break

    # Renvoyer les statistiques d'entrainement
    return history

def evaluate(model, test_loader, encoder, criterion):
    # Variables pour les statistiques
    correct = []
    actual_labels = []
    total_loss = 0.0 
    # Passer le modèle en mode évaluation
    model.eval()
    # Désactiver le calcul du gradient
    with torch.no_grad():
        # Pour chaque batch du loader
        for batch in test_loader:
            # Récupérer les informations du batch
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['label']
            # Forward pass au modèle
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            # Calcul de la loss
            loss = criterion(outputs, labels)
            # Incrémenter les statistiques
            total_loss += loss.item()
            correct.extend(torch.argmax(outputs, 1).cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
    # Afficher les statistiques
    avg_loss = total_loss / len(test_loader)
    accuracy = accuracy_score(actual_labels, correct)
    print(f"Accuracy Score: {accuracy:.4f} - Loss: {avg_loss:.4f}")
    # On décode les labels
    actual_labels = encoder.inverse_transform(actual_labels)
    correct = encoder.inverse_transform(correct)
    # Afficher un classification report et une matrice de confusion
    print(f"Classification report:\n {classification_report(actual_labels, correct)}")
    ConfusionMatrixDisplay.from_predictions(actual_labels, correct)
    plt.show()