# Lingua
Biz Lingua takımı olarak doğal dil işleme alanında çalışıyoruz. Amacımız, Türkçe doğal dil işleme kaynaklarını geliştirmek ve Teknofest gibi yarışmalarda başarılı olmaktır.

# Confusion Matrix

Confusion Matrix şu kod kullanılarak hesaplanmıştır.

```python 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


df = pd.read_csv("confusion_matrix_icin.csv", sep="|")
y_true = df['normal_target'].tolist()
y_pred = df['predicted_target'].tolist()


cm = confusion_matrix(y_true, y_pred)
class_names = ['OTHER', 'SEXIST','RACIST','INSULT','PROFANITY']
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))

plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names) 
thresh = cm_norm.max() / 2.

for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd') + "\n({:.2f})".format(cm_norm[i, j]), horizontalalignment="center", color="white" if cm_norm[i, j] > thresh else "black")
             
plt.tight_layout()
plt.ylabel('Real Class')
plt.xlabel('Predicted Class')
plt.show()
```

![indir (1)](https://user-images.githubusercontent.com/81961593/230114000-7a518281-9674-4267-96b4-e3ac9c9d772b.png)

# F1 Skoru

F1 Skoru, aşağıdaki kod kullanılarak hesaplanmıştır ve "0.95" değeri bulunmuştur.

```python 
from sklearn.metrics import f1_score
import pandas as pd

df = pd.read_csv("confusion_matrix_icin.csv", sep="|")
y_true = df['normal_target'].tolist()
y_pred = df['predicted_target'].tolist()

f1 = f1_score(y_true, y_pred, average='weighted')
print("F1 skoru:", f1)
```

<img width="1072" alt="image" src="https://user-images.githubusercontent.com/81961593/230116788-2714a0ee-4a49-45b6-a507-cb6d4b5fb653.png">
