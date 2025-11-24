# Brain Tumor Classification - ResNet18 Fine-tuning

Bu proje, beyin tümörlerini sınıflandırmak için **ResNet18** modelini fine-tuning yaparak eğitir.

## 📚 Öğrenme Amaçlı Proje

Bu proje, fine-tuning sürecini **adım adım öğrenmek** için hazırlanmıştır. Her dosya detaylı açıklamalar içerir.

## 🗂️ Proje Yapısı

```
recruitment_case_study/
├── data/
│   ├── Training/          # Eğitim verileri
│   │   ├── glioma_tumor/
│   │   ├── meningioma_tumor/
│   │   ├── no_tumor/
│   │   └── pituitary_tumor/
│   └── Testing/           # Test verileri
├── models/
│   └── resnet18.py        # Model tanımı
├── src/
│   ├── dataset.py         # Veri yükleme ve augmentation
│   ├── model.py           # Model oluşturma ve freeze stratejileri
│   ├── train.py           # Training ve validation loop'ları
│   ├── main.py            # Ana training script
│   └── inference.py       # Tahmin yapma
├── requirements.txt       # Gerekli kütüphaneler
└── README.md             # Bu dosya
```

## 🚀 Kurulum

### 1. Gerekli Kütüphaneleri Yükle

```bash
pip install -r requirements.txt
```

### 2. Veri Setini Hazırla

Veri setiniz şu formatta olmalı:
```
data/
  Training/
    class1/
      image1.jpg
      image2.jpg
    class2/
      ...
  Testing/
    class1/
      ...
```

## 📖 Dosyaları Öğrenme Sırası

### 1. `src/dataset.py` - Veri Yükleme
**Ne öğrenirsin?**
- `ImageFolder` ile veri yükleme
- Data Augmentation teknikleri
- `DataLoader` kullanımı
- Train/Validation split

**Çalıştır:**
```bash
cd src
python dataset.py
```

### 2. `src/model.py` - Model Hazırlama
**Ne öğrenirsin?**
- Pretrained model yükleme
- Fine-tuning stratejileri (freeze/unfreeze)
- Learning rate ayarları
- Optimizer ve Scheduler

**Çalıştır:**
```bash
python model.py
```

### 3. `src/train.py` - Training Loop
**Ne öğrenirsin?**
- Training loop nasıl yazılır
- Forward/Backward pass
- Gradient descent
- Validation
- Model kaydetme

**Not:** Bu dosya doğrudan çalıştırılmaz, main.py içinde kullanılır.

### 4. `src/main.py` - Ana Training
**Ne öğrenirsin?**
- Tüm componentleri birleştirme
- Hyperparameter ayarlama
- Training süreci yönetimi

**Çalıştır:**
```bash
python main.py
```

### 5. `src/inference.py` - Tahmin Yapma
**Ne öğrenirsin?**
- Eğitilmiş model yükleme
- Yeni veriler üzerinde tahmin
- Sonuç görselleştirme

**Çalıştır:**
```bash
python inference.py
```

## 🎯 Hızlı Başlangıç

Tüm süreci başlatmak için:

```bash
# 1. Kütüphaneleri yükle
pip install -r requirements.txt

# 2. Training başlat
cd src
python main.py
```

## ⚙️ Hyperparameter Ayarlama

`src/main.py` dosyasındaki `config` dictionary'sini düzenle:

```python
config = {
    'batch_size': 32,              # GPU memory'e göre: 16, 32, 64
    'num_epochs': 25,              # Epoch sayısı
    'learning_rate': 0.001,        # Learning rate
    'freeze_strategy': 'partial',  # 'none', 'all', 'partial'
    'val_split': 0.2,             # Validation oranı
}
```

### Freeze Stratejileri

1. **`'all'` - Feature Extraction**
   - Tüm pretrained layer'lar donmuş
   - Sadece son layer eğitilir
   - Hızlı, az veri için iyi

2. **`'partial'` - Kademeli Fine-tuning (Önerilen)**
   - İlk layer'lar donmuş
   - Son layer'lar eğitilir
   - Denge sağlar

3. **`'none'` - Full Fine-tuning**
   - Tüm model eğitilir
   - En yavaş, çok veri gerektirir
   - En iyi sonuçları verebilir

## 📊 Training Sonrası

Training bittikten sonra:

1. **Model:** `models/resnet18_tumor_classifier.pth`
2. **Grafik:** `models/training_history.png`
3. **Metrics:** Console'da yazdırılır

## 🔍 İnference (Tahmin)

```python
from inference import predict_single_image, load_trained_model

# Model yükle
model = load_trained_model('models/resnet18_tumor_classifier.pth')

# Tahmin yap
pred, conf, probs = predict_single_image(
    model, 
    'path/to/image.jpg',
    ['glioma', 'meningioma', 'no_tumor', 'pituitary']
)

print(f"Tahmin: {pred}, Güven: {conf:.2%}")
```

## 📚 Kavramlar Sözlüğü

### Fine-tuning Nedir?
Pretrained bir modeli kendi veri setinizde tekrar eğitme sürecidir. Model önceden öğrendiklerini kullanarak yeni görevde daha hızlı öğrenir.

### Freezing/Unfreezing
- **Freeze:** Layer'ı dondurma, weight'leri güncelleme
- **Unfreeze:** Layer'ı eğitilebilir yapma

### Learning Rate
Model'in her adımda ne kadar değişeceği. Çok büyük: kararsız, çok küçük: yavaş öğrenme.

### Data Augmentation
Veriyi yapay olarak çeşitlendirme (döndürme, çevirme, renk değiştirme). Overfitting'i azaltır.

### Batch Size
Bir anda işlenen görsel sayısı. GPU memory'e bağlı.

### Epoch
Tüm veri setinin bir kez görülmesi.

## 🎓 Sonraki Adımlar

1. ✅ Farklı freeze stratejileri dene
2. ✅ Hyperparameter'ları optimize et
3. ✅ Daha fazla augmentation ekle
4. ✅ Farklı model mimarileri dene (ResNet50, EfficientNet)
5. ✅ Ensemble yöntemleri öğren

## 🐛 Yaygın Hatalar ve Çözümler

### CUDA Out of Memory
```python
# Batch size'ı küçült
config['batch_size'] = 16
```

### Overfitting
```python
# Daha fazla augmentation ekle
# Dropout ekle
# Regularization kullan
```

### Underfitting
```python
# Daha fazla epoch
# Learning rate artır
# Daha az freeze
```

## 📝 Notlar

- Her dosya standalone çalışabilir (test amaçlı)
- Detaylı açıklamalar kod içinde
- Her fonksiyon docstring içerir
- Öğrenme amaçlı yazılmıştır

## 🤝 Katkı

Bu proje eğitim amaçlıdır. Sorularınız için issue açabilirsiniz.

## 📄 Lisans

MIT License

---

**Happy Learning! 🚀**
