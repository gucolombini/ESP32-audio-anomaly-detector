#!/usr/bin/env python3
import json, random
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from features import load_wav, mfcc

SEED=42; random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)
items=[]
for label,name in enumerate(("background","cough")):
    for p in (Path("data")/name).glob("*.wav"): items.append((p,label))
if min(sum(y==k for _,y in items) for k in (0,1))<20:
    raise SystemExit("Colete pelo menos 20 WAVs por classe em data/{background,cough}")
items.sort(key=lambda item:str(item[0])); X=np.stack([mfcc(load_wav(p)).reshape(-1) for p,_ in items]); y=np.array([y for _,y in items])
# Split estratificado por arquivo. Para maior rigor, reserve uma sessão/pessoa inteira.
Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,test_size=.30,random_state=SEED,stratify=y)
Xv,Xte,yv,yte=train_test_split(Xtmp,ytmp,test_size=.50,random_state=SEED,stratify=ytmp)
mean=Xtr.mean(axis=0); std=np.maximum(Xtr.std(axis=0),1e-3)
Xtrn=(Xtr-mean)/std; Xvn=(Xv-mean)/std; Xten=(Xte-mean)/std
class_weight={k:len(ytr)/(2*int(np.sum(ytr==k))) for k in (0,1)}
model=tf.keras.Sequential([tf.keras.Input((49*13,)),
    tf.keras.layers.Dense(32,activation="relu"),tf.keras.layers.Dropout(.25),
    tf.keras.layers.Dense(2,activation="softmax")])
model.compile("adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
cb=[tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True)]
model.fit(Xtrn,ytr,validation_data=(Xvn,yv),epochs=120,batch_size=16,callbacks=cb,
          class_weight=class_weight,verbose=2)
# 0,5 preserva a regra natural do classificador binário. Com poucas tosses, um
# limiar otimizado em apenas uma divisão varia demais e tende a sobreajustar.
threshold=.5
test_prob=model.predict(Xten,verbose=0)[:,1]; pred=(test_prob>=threshold).astype(int)

# Incorpora a normalização na primeira camada: (x-mean)/std @ W + b.
# Assim o firmware recebe MFCC bruto e continua precisando apenas de FullyConnected.
first=next(layer for layer in model.layers if isinstance(layer,tf.keras.layers.Dense))
w,b=first.get_weights(); first.set_weights([w/std[:,None],b-(mean/std)@w])
Path("models").mkdir(exist_ok=True)
model.export("models/saved_model")
metrics={"confusion_matrix":confusion_matrix(yte,pred).tolist(),
         "report":classification_report(yte,pred,target_names=["background","cough"],output_dict=True),
         "threshold":threshold,"parameters":model.count_params(),
         "samples":{"train":len(ytr),"validation":len(yv),"test":len(yte)}}
Path("models/metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
print(json.dumps(metrics,indent=2))
