# BERT-distilBERT-sentiment-analysis

# Sentiment Analysis BERT & DistilBERT

## Project Overview

Το παρόν project υλοποιεί ένα σύστημα **binary sentiment classification** για tweets (θετικό / αρνητικό συναίσθημα), αξιοποιώντας τις αρχιτεκτονικές:

- BERT
- DistilBERT

Η προσέγγιση βασίζεται σε **fine-tuning προεκπαιδευμένων Transformer μοντέλων**, με εκτενή πειραματισμό σε υπερπαραμέτρους και τεχνικές regularization.

---

## Repository Structure

Το repository περιλαμβάνει δύο βασικά notebooks:

- `Bert_sentiment_an.ipynb`
- `distilBERT_sentiment_an.ipynb`

🔹 **Σημαντική σημείωση:**  
Σε κάθε notebook, **το πρώτο block περιέχει ολόκληρο τον βασικό κώδικα υλοποίησης** (ορισμοί 
συναρτήσεων, training pipeline, evaluation pipeline).  
Τα επόμενα blocks εκτελούν τα επιμέρους πειράματα αλλάζοντας υπερπαραμέτρους, σύμφωνα με το 
report και για τους δύο transformers.

---

## Data Processing

### Preprocessing

Η προεπεξεργασία σχεδιάστηκε ώστε να διατηρείται η σημασιολογική πληροφορία, η οποία είναι κρίσιμη για Transformer models:

- Lowercasing
- Expansion αγγλικών contractions (π.χ. *don’t → do not*)
- Normalization επαναλαμβανόμενων χαρακτήρων (π.χ. *soooo → soo*)
- Διατήρηση:
  - Hashtags
  - Mentions
  - Emojis
  - Σημείων στίξης

Δεν εφαρμόστηκε αφαίρεση stopwords ή stemming, καθώς τα BERT-based μοντέλα αξιοποιούν την πλήρη μορφολογία του κειμένου.

---

### Tokenization & Vectorization

Χρησιμοποιήθηκαν οι προκαθορισμένοι tokenizers:

- `bert-base-uncased`
- `distilbert-base-uncased`

Χαρακτηριστικά:
- Padding & truncation
- Μέγιστο μήκος ακολουθίας: **40 tokens** (επιλογή βάσει στατιστικής ανάλυσης του μήκους tweets)
- Δημιουργία:
  - `input_ids`
  - `attention_mask`

Δεν έγινε εξωτερική επεξεργασία embeddings — η προσαρμογή έγινε μέσω fine-tuning.

---

# Προσέγγιση πειραμάτων

Όλα τα πειράματα εκτελέστηκαν για **3 epochs**.  
Η αξιολόγηση κάθε trial βασίστηκε στο **epoch με το καλύτερο validation score** (συχνά το 2ο epoch).

Χρησιμοποιήθηκαν οι εξής μετρικές:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC Curve
- Learning Curves

---

# Πειράματα BERT

Η πειραματική διαδικασία οργανώθηκε σε στάδια:

##  Batch Size & Learning Rate Exploration

Δοκιμάστηκαν:
- Batch sizes: 16, 32, 64, 96
- Learning rates: 1e-5, 2e-5, 4e-5, 1e-4

Συμπεράσματα:
- Μεγαλύτερο batch size (96) → πιο σταθερή εκπαίδευση
- Learning rate = **4e-5** → καλύτερο validation F1
- Learning rate = 1e-4 → αποσταθεροποίηση

---

##  Regularization & Dropout Tuning

Χρησιμοποιήθηκε **Optuna** για εύρεση καλών αρχικών τιμών:

- Weight decay
- Hidden dropout
- Attention dropout
- Classification dropout

Διαπιστώθηκε ότι:
- Dropout ≈ 0.15 λειτουργεί ισορροπημένα
- Υπερβολικό dropout → υποβάθμιση απόδοσης
- Πολύ χαμηλό regularization → τάση για overfitting

---

##  3. Σύγκριση Scheduler

Δοκιμάστηκαν:

- Linear Warmup
- Cosine Annealing with Warm Restarts (CAWR)
- ReduceLROnPlateau

Το **Linear Warmup** παρέμεινε η πιο σταθερή επιλογή.  
Οι εναλλακτικοί schedulers δεν προσέφεραν βελτίωση.

---

## Καλύτερο configuration για BERT

- Learning rate: 4e-5
- Batch size: 96
- Weight decay: ~0.06–0.15
- Dropouts: ~0.15
- Scheduler: Linear Warmup

Το μοντέλο παρουσίασε ισχυρή επίδοση στο validation με καλή ισορροπία bias–variance.

---

# Πειράματα distilBERT

Το DistilBERT ακολούθησε αντίστοιχη διαδικασία, με ορισμένες διαφοροποιήσεις.

## 1. Αρχικές τροποποιήσεις

- Batch size: 16 απέδωσε καλύτερα
- Learning rate: 2e-5 ή 1e-5 πιο σταθερά
- Υψηλό learning rate → έντονο overfitting

---

## 2. Optuna-Based Regularization

Προτάθηκαν τιμές:

- Weight decay: 0.25
- Hidden dropout: 0.3
- Attention dropout: 0.07
- Classifier dropout: 0.001

Με αύξηση του weight decay σε **0.5**, παρατηρήθηκε μικρή βελτίωση στη γενίκευση.

---

## 3. Σύγκριση Scheduler

Όπως και στο BERT:
- Linear Warmup παρέμεινε η πιο αξιόπιστη επιλογή.
- CAWR & ReduceLROnPlateau δεν έδωσαν καλύτερα αποτελέσματα.

---

#  Σύγκριση BERT και distilBERT

| Model       | Stability | Best Validation Performance | Computational Cost |
|------------|-----------|----------------------------|--------------------|
| BERT       | Υψηλή     | Ελαφρώς καλύτερη          | Μεγαλύτερο         |
| DistilBERT | Πολύ καλή | Οριακά χαμηλότερη         | Μικρότερο          |

### Συμπέρασμα

- Το **BERT** πέτυχε την καλύτερη συνολική επίδοση.
- Το **DistilBERT** προσφέρει πολύ καλή απόδοση με μικρότερο υπολογιστικό κόστος.
- Το fine-tuning των υπερπαραμέτρων (ιδίως learning rate και regularization) είχε σημαντικότερο αντίκτυπο από την αλλαγή scheduler.

---

