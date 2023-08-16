# Lingua
![indir (1)](https://avatars.githubusercontent.com/u/129843671?s=400&u=54b132a4bbd3fe1822279ae20dcc4080bc98fabd&v=4)

Merhaba! Biz Lingua takımı olarak, doğal dil işleme alanında çalışıyoruz ve Türkçe doğal dil işleme kaynaklarını geliştirmek için yoğun bir şekilde çaba gösteriyoruz. Amacımız sadece bununla da sınırlı değil aynı zamanda Teknofest gibi prestijli yarışmalarda da başarılı olmak istiyoruz. Kendimize özgü yaklaşımımız ve yaratıcı çözümlerimizle, doğal dil işleme alanında yeni bir çağ açmayı hedefliyoruz.
# Model

Modelimiz Türkçe metin sınıflandırması için özellikle tasarlanmıştır ve Türkçe BERT adı verilen bir yapay zeka modeli kullanılarak eğitilmiştir. Bu model, Türkçe metinlerdeki kelime ve cümle yapılarını anlamak için tasarlanmış bir dil modelidir ve doğal dil işleme alanındaki birçok farklı görev için kullanılabilir.

Özellikle metin sınıflandırması alanında kullanılan bu model, belirli bir metnin hangi sınıfa ait olduğunu belirlemek için kullanılabilir. Örneğin, bir tweet'in "insult" (hakaret) veya "racist" (ırkçılık) olarak sınıflandırılması gibi.

Türkçe BERT modelimiz, büyük bir Türkçe metin veri kümesi kullanılarak eğitildi ve sonuç olarak, Türkçe doğal dil işleme alanında oldukça başarılı sonuçlar vermektedir. Ayrıca, modelimizin girdi olarak alabileceği metin boyutu oldukça geniş olduğundan, farklı uzunluklardaki metinler için de kullanılabilir.

Bu modelin özellikleri arasında, doğal dil işleme alanındaki diğer görevler için de kullanılabilmesi ve birçok farklı Türkçe veri kümesi üzerinde yüksek doğruluk oranlarına sahip olması yer almaktadır.
# Performans

Modelimiz, %95 doğruluk elde etti. Ayrıca, 0.95 F1 puanı elde edildi.

Modelimizi eğitmek için 12.000 adet cümle içeren veri seti kullanıldı ve eğitim süresi yaklaşık 3.5 saat sürdü.

# Veri Seti

Modelimizi eğitmek için TEKNOFEST'in verdiği veri setini kullandık. Bu veri seti, 12.000 adet hakaret, küfür, ırkçılık, cinsiyetçilik ve herhangi bir kategoriye girmeyen cümleler içeriyor ve her bir cümle, ayrı bir etiketle etiketlenmiştir. Etiketler, 5 farklı sınıfa ayrılmıştır: OTHER, SEXIST, RACIST, INSULT ve PROFANITY.

# Modeli İndirme

Modelimiz yaklaşık 4 GB olduğundan dolayı Github'a ekleyemedik. Modelimizi aşağıda verilen tablodaki linklerden birini kullanarak indirebiriliz.

Sağlayıcı | Link
--- | --- |
Google Drive | [Google Drive](https://drive.google.com/file/d/1NK6ZyTdQo73uZZ34QzPdvDwk7DXSaZnZ/view?usp=sharing) |

# Modelin Kullanımı
Model için kurmanız gereken kütüphane
```python
pip install simpletransformers
```
Modeli ayrıca kullanmak isterseniz aşağıdaki gibi kullanabilirsiniz.

```python
from simpletransformers.classification import ClassificationModel
def predict(texts):
    model_path = "bert_model"
    model = ClassificationModel('bert', model_path, use_cuda=False)
    predictions, _ = model.predict(texts)
    return [sayidan_sonuca(prediction) for prediction in predictions]


def sayidan_sonuca(sayi):
    if sayi == 4:
        return 'OTHER'
    elif sayi == 1:
        return 'RACIST'
    elif sayi == 0:
        return 'INSULT'
    elif sayi == 3:
        return 'PROFANITY'
    elif sayi == 2:
        return 'SEXIST'
        
print(predict([""]))   #Sınıflandıralacak Metin Buraya girilecek.
```
# Confusion Matrix

Confusion Matrix şu kod kullanılarak hesaplanmıştır ve aşağıda verilen resim gelmiştir.

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

# Demo Videosu

https://user-images.githubusercontent.com/81961593/232031636-069fdd2b-3cdb-45df-8977-d81ef38ed013.mp4

Videoyu Youtube üzerinden izlemek için aşağıdaki linki kullanabilirsiniz:

https://www.youtube.com/watch?v=IpdUwnNhR50
