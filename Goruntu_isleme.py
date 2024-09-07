# -*- coding: utf-8 -*-
"""
Created on Sun May 26 10:59:29 2024

@author: AdemTrks
"""

import tensorflow

#özellik çıkarma işleminde evrişim ve havuzlama işlemleri ile gerçekleşiyor.
#bu evrişim ve havuzlama işlemide daha önceden yazılmış transfer öğrenme ile bu işlemi yapıyor.
OZELLIK_CIKARAN_MODEL=tensorflow.keras.applications.VGG16(
    weights='imagenet',# ImageNet üzerinde eğitilmiş ağırlıklar
    include_top=False,
    input_shape=(224,224,3))#3 RGB


OZELLIK_CIKARAN_MODEL.summary()#özet
#convd2d (evrişim)  #maxpooling2d(havuzlama) katmanı işlemleri yapıyor.
#amaç giriş görüntülerinin boyutunu düşürmek sonunda (7*7 lik matrislerden oluşan 512 tane matris oluşturuyor.)


OZELLIK_CIKARAN_MODEL.trainable=True
set_trainable=True
for layer in OZELLIK_CIKARAN_MODEL.layers:
    if layer.name=='block5_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainable=True
    else:
        layer.trainable=False

#bu kod çıkarılmasına gerek yok ama tekrardan yapılırsa eğitim sırasında bu bölümü tekrardan eğiticek
#ve bilgisayarı ektstra eğiticek.


#boş model oluşturuyor
model=tensorflow.keras.models.Sequential()
 
#tensorflowda Sequential methodu işlemlerin sıralı yapılacağını belirtiyor.


model.add(OZELLIK_CIKARAN_MODEL)


#girdi verilerini düzleştiren bir işlevdir.
#girdiyi tek boyutlu bir vektöre dönüştürür
model.add(tensorflow.keras.layers.Flatten())


#oluşturduğumuz 7*7*512 matrisin sonuna önce 256 nöron ekliyor ve ardından
#2 sınıflı bir sınıflandırma yapacağımız için 2 adet de onu ekliyoruz.

model.add(tensorflow.keras.layers.Dense(256,activation='relu'))
model.add(tensorflow.keras.layers.Dense(2,activation='softmax'))


#ReLU: ReLU, sinir ağlarındaki gizli katmanlarda sıkça kullanılan bir aktivasyon fonksiyonudur çünkü hızlı ve etkili öğrenmeyi destekler.
#Softmax: Softmax, modelin çıktısının bir olasılık dağılımı olmasını sağlar ve bu nedenle sınıflandırma problemlerinde özellikle son katmanda kullanılır.


#modeli derleme
model.compile(loss='binary_crossentropy',
              optimizer=tensorflow.keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=['acc'])



#görüntülerimizin olduğu klasörlerin yolunu yazıyoruz.
EGITIM_YOLU='EGITIM'
GECERLEME_YOLU='GECERLEME'
TEST_YOLU='TEST'

#aşırı öğrenmenin önüne geçmek için veri arttırma yöntemleri kullanmamız gerekiyor.
#keras ın görüntü ön işleme modelini çağırıyor.


train_datagen = tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.255, #piksel değerleri 0-255 den 0-1 arasına getiriliyor
    rotation_range=40, #istenilen arttırılma işlemleri yapılabilir.
    width_shift_range=0.2,#sağ sol döndürme,aşağı yukarı çevirme,zoom yapma
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

#yukarıda oluşturduğumuz görüntü çoğaltma işlemini Eğitim klasöründeki verilerimize uyguluyoruz.
train_generator=train_datagen.flow_from_directory(
   EGITIM_YOLU,
   target_size=(224,224),
   batch_size=16,#her seferinde 16 adet görüntü okuyacak.
   )


validation_datagen=tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
#oluşturduğu geçerleme kodunu uyguluyor
validation_generator=validation_datagen.flow_from_directory(
    GECERLEME_YOLU,
    target_size=(224,224),
    batch_size=16,#her seferinde 16 adet görüntü okuyacak.
    )

#CHECKPOINT
#MODELLOT
#eğitimi gerçekleştiricek komut
EGITIM_TAKIP=model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=5
    )

#modelin kaydedilicek adı.
model.save('50_epoch_LEKELI_PENCE_AYIRAN_MODEL.h5')

#test görüntüleri içinde orjinelliği bozmuyor sadece 01 arasına getirmek için 255 bölüyor.
test_datagen=tensorflow.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255
    )
#oluşturduğu test kodunu uyguluyor
test_generator=test_datagen.flow_from_directory(
    TEST_YOLU,
    target_size=(224,224),
    batch_size=16,
    )

#doğruluğu ekrana yazdırıyor.
test_loss, test_acc = model.evaluate(test_generator, steps=50)
print('test acc:',test_acc)

 