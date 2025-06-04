## [반도체 결함 탐지 모델](#반도체-결함-탐지-모델)
```python
!pip install tensorflow
```


```python
import pandas as pd
import numpy as np
import torch
import tensorflow as tf
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM as OCS
from tqdm import tqdm
from pyod.models.abod import ABOD
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers
```


```python
# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```


```python
# 데이터 로딩 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): csv 파일의 경로.
            transform (callable, optional): 샘플에 적용될 Optional transform.
        """
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df['img_path'].iloc[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

# 이미지 전처리 및 임베딩
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = CustomDataset(csv_file='./train.csv', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=False)
```


```python
# 사전 학습된 모델 로드
model = models.resnet50(pretrained=True)
model.eval()  # 추론 모드로 설정

# 특성 추출을 위한 모델의 마지막 레이어 수정
model = torch.nn.Sequential(*(list(model.children())[:-1]))

model.to(device)

# 이미지를 임베딩 벡터로 변환
def get_embeddings(dataloader, model):
    embeddings = []
    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(device)
            emb = model(images)
            embeddings.append(emb.cpu().numpy().squeeze())
    return np.concatenate(embeddings, axis=0)

train_embeddings = get_embeddings(train_loader, model)
```


```python
# Isolation Forest 모델 학습
clf = IsolationForest(random_state=42)
clf.fit(train_embeddings)
```


```python
# 테스트 데이터에 대해 이상 탐지 수행
test_data = CustomDataset(csv_file='./test.csv', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

test_embeddings = get_embeddings(test_loader, model)
test_pred = clf.predict(test_embeddings)

# Isolation Forest의 예측 결과(이상 = -1, 정상 = 1)를 이상 = 1, 정상 = 0으로 변환
test_pred = np.where(test_pred == -1, 1, 0)
```


```python
from bayes_opt import BayesianOptimization
```


```python
def ocs_eval(nu):
    model = OCS(
        kernel = 'rbf',
        nu=float(nu),
        gamma='auto',
        )
    model.fit(train_embeddings)
    label = model.predict(train_embeddings)
    label = np.where(label == -1, 1, 0)
    try:
        score = silhouette_score(train_embeddings, label)
    except ValueError:
        score = -1.0
    return score

# 하이퍼파라미터 범위 설정
pbounds = {
    'nu': (0.0000001, 0.2)
}

optimizer = BayesianOptimization(f=ocs_eval, pbounds=pbounds, random_state = 42)
optimizer.maximize(init_points=3, n_iter=20)
```


```python
max_para = optimizer.max['params']
```


```python
ocs = OCS(kernel='rbf',
          nu=max_para['nu'], 
          gamma = 'auto'
         ).fit(train_embeddings)

test_pred = ocs.predict(test_embeddings)
test_pred = np.where(test_pred == -1, 1, 0)
```


```python
submit = pd.read_csv('./sample_submission.csv')
submit['label'] = test_pred

submit.to_csv('./result_submit.csv', index=False)
```


```python
from sklearn.model_selection import train_test_split
import keras
```


```python
X_train, X_test = train_test_split(train_embeddings, test_size=0.2, random_state=42)
```


```python
input_dim = X_train.shape[1]

encoder = keras.models.Sequential([
    keras.layers.Dense(400, activation='relu', input_shape=[input_dim]),
    keras.layers.Dropout(rate=0.1),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(rate=0.1),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dropout(rate=0.1),
    keras.layers.Dense(10, activation='relu')
])

decoder = keras.models.Sequential([
    keras.layers.Dense(50, activation='relu', input_shape=[10]),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dropout(rate=0.1),
    keras.layers.Dense(400, activation='relu'),
    keras.layers.Dropout(rate=0.1),
    keras.layers.Dense(input_dim, activation='relu'),
])

autoencoder = keras.models.Sequential([encoder, decoder])

autoencoder.compile(
    loss='mean_squared_error', optimizer=keras.optimizers.Adam(), metrics=['mse'])
```


```python
nb_epoch = 100
batch_size = 320
autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="model.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_test, X_test),
                    verbose=1,
                    callbacks=[checkpointer, tensorboard]).history
```


```python
autoencoder = load_model('model.h5')
```


```python
pred = autoencoder.predict(test_embeddings)
```


```python
threshold = history['loss'][max(history['accuracy']) == history['accuracy']]

loss = [0]*(test_embeddings.shape[0])

for i in range(test_embeddings.shape[0]):
        loss[i] = sum((test_embeddings[i] - pred[i])**2) / test_embeddings.shape[1]

test_pred = np.array(loss) > threshold

test_pred = test_pred.astype(int)
```


```python
test_pred
```