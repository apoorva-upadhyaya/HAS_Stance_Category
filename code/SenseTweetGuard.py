import tensorflow as tf
from keras import backend as K
from keras import backend as k
from keras import layers
from keras.layers.core import Lambda
import numpy as np
from numpy import array 
import pandas as pd
from keras.preprocessing.text import Tokenizer
from  keras.utils import pad_sequences
from keras.layers import LSTM,Dropout,Bidirectional,Input, Embedding, Dense,Concatenate,Flatten, Multiply,Average,Subtract,GRU
from keras.models import Model
from tensorflow.python.keras.layers.merge import concatenate
# from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
from keras.utils import to_categorical
from keras.optimizers import Adam
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from keras import optimizers,regularizers
from sklearn.utils import resample
from sklearn.metrics import multilabel_confusion_matrix
import tensorflow_probability as tfp
import flipGradientTF
from transformers import TFAutoModel, AutoTokenizer


def gradRev(x):
    Flip = flipGradientTF.GradientReversal(1)
    x = Flip(x)
    return x

def correlatedFeatures(var):
    ten1,ten2=var[0],var[1]
    arr1=tfp.stats.correlation(ten1, ten2,sample_axis=-1,event_axis=None)
    arr1=k.expand_dims(arr1, 1)
    return arr1

def attentionScores(var):
    t1,t2,t3=var[0],var[1],var[2]
    Q_t=t1
    K_t=t2
    V_t=t3
    scores = tf.matmul(Q_t, K_t, transpose_b=True)
    distribution = tf.nn.softmax(scores)
    scores=tf.matmul(distribution, V_t)
    return scores


data=pd.read_csv("../web/train.csv", delimiter="\t", na_filter= False) 
print("toxic data :: ",len(data))

train_tid=[]
for i in range(len(data)):
    train_tid.append(str(data.tweet_id.values[i]))

data=pd.read_csv("../web/test.csv", delimiter="\t", na_filter= False) 
print("toxic data :: ",len(data))

test_tid=[]
for i in range(len(data)):
    test_tid.append(str(data.tweet_id.values[i]))

data=pd.read_csv("../web/val.csv", delimiter="\t", na_filter= False) 
print("toxic data :: ",len(data))

val_tid=[]
for i in range(len(data)):
    val_tid.append(str(data.tweet_id.values[i]))


data=pd.read_csv("../data/final.csv", delimiter=";", na_filter= False) 
print("data :: ",len(data))


########### creating multiple lists as per the daataframe

li_text_train=[]
li_stance_train=[]
li_sent_train=[]
li_id_train=[]
li_hate_train=[]
li_topic_train=[]

li_text_test=[]
li_stance_test=[]
li_sent_test=[]
li_id_test=[]
li_hate_test=[]
li_topic_test=[]

li_text_val=[]
li_stance_val=[]
li_sent_val=[]
li_id_val=[]
li_hate_val=[]
li_topic_val=[]

for i in range(len(data)):
    id_=str(data.tweetid.values[i])
    if id_ in train_tid:
        li_id_train.append(data.tweetid.values[i])
        li_stance_train.append(str(data.stance.values[i]))
        li_text_train.append(str(data.text.values[i]))
        li_sent_train.append(str(data.sentiment.values[i]))
        li_hate_train.append(str(data.hate.values[i]))
        li_topic_train.append(str(data.topic.values[i]))

    elif id_ in test_tid:
        li_id_test.append(data.tweetid.values[i])
        li_stance_test.append(str(data.stance.values[i]))
        li_text_test.append(str(data.text.values[i]))
        li_sent_test.append(str(data.sentiment.values[i]))
        li_hate_test.append(str(data.hate.values[i]))
        li_topic_test.append(str(data.topic.values[i]))

    else:
        li_id_val.append(data.tweetid.values[i])
        li_stance_val.append(str(data.stance.values[i]))
        li_text_val.append(str(data.text.values[i]))
        li_sent_val.append(str(data.sentiment.values[i]))
        li_hate_val.append(str(data.hate.values[i]))
        li_topic_val.append(str(data.topic.values[i]))


print("len of train::",len(li_stance_train))
print("len of test::",len(li_stance_val))
print("len of val::",len(li_stance_val))

print("li_stance np unique:::",np.unique(li_stance_train,return_counts=True))
print("li_sent np unique:::",np.unique(li_sent_train,return_counts=True))
print("li_hate np unique:::",np.unique(li_hate_train,return_counts=True))
print("li_topic np unique:::",np.unique(li_topic_train,return_counts=True))

list_acc_stance,list_f1_stance,list_prec_stance,list_rec_stance=[],[],[],[]
list_acc_topic,list_f1_topic,list_prec_topic,list_rec_topic=[],[],[],[]
list_acc_hate,list_f1_hate,list_prec_hate,list_rec_hate=[],[],[],[]
list_acc_sent,list_f1_sent,list_prec_sent,list_rec_sent=[],[],[],[]

########### converting labels into categorical labels ########
label_encoder=LabelEncoder()
final_lbls=li_stance_train+li_stance_val+li_stance_test
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)
train_stance_enc=total_integer_encoded[:len(li_stance_train)]
val_stance_enc=total_integer_encoded[len(li_stance_train):len(li_stance_train)+len(li_stance_val)]
test_stance_enc=total_integer_encoded[len(li_stance_train)+len(li_stance_val):]

######### hate ##########
label_encoder=LabelEncoder()
final_lbls=li_hate_train+li_hate_val+li_hate_test
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)
train_hate_enc=total_integer_encoded[:len(li_stance_train)]
val_hate_enc=total_integer_encoded[len(li_stance_train):len(li_stance_train)+len(li_stance_val)]
test_hate_enc=total_integer_encoded[len(li_stance_train)+len(li_stance_val):]

######### topic ##########
label_encoder=LabelEncoder()
final_lbls=li_topic_train+li_topic_val+li_topic_test
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)
train_topic_enc=total_integer_encoded[:len(li_stance_train)]
val_topic_enc=total_integer_encoded[len(li_stance_train):len(li_stance_train)+len(li_stance_val)]
test_topic_enc=total_integer_encoded[len(li_stance_train)+len(li_stance_val):]

######### sentiment ##########
label_encoder=LabelEncoder()
final_lbls=li_sent_train+li_sent_val+li_sent_test
values=array(final_lbls)
total_integer_encoded=label_encoder.fit_transform(values)

train_sent_enc=total_integer_encoded[:len(li_stance_train)]
val_sent_enc=total_integer_encoded[len(li_stance_train):len(li_stance_train)+len(li_stance_val)]
test_sent_enc=total_integer_encoded[len(li_stance_train)+len(li_stance_val):]

print("len of train, test, val*******************",len(train_sent_enc),len(test_sent_enc),len(val_sent_enc))

# Define the tokenizer
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
train_encodings = tokenizer(li_text_train, padding=True, truncation=True, return_tensors="tf")
val_encodings = tokenizer(li_text_val, padding=True, truncation=True, return_tensors="tf")
test_encodings = tokenizer(li_text_test, padding=True, truncation=True, return_tensors="tf")
bertweet_model = TFAutoModel.from_pretrained("vinai/bertweet-base")
bertweet_model.trainable = False

#### stance #######
input_ids1 = Input(shape=(None,), dtype='int32', name='input_ids1')
attention_mask1 = Input(shape=(None,), dtype='int32', name='attention_mask1')
outputs = bertweet_model(input_ids=input_ids1, attention_mask=attention_mask1)
sequence_output = outputs.pooler_output
Q_s= Dense(100, activation="relu")(sequence_output)
K_s= Dense(100, activation="relu")(sequence_output)
V_s= Dense(100, activation="relu")(sequence_output)
IA_stance=Lambda(attentionScores)([Q_s,K_s,V_s])
stance_specific_output=Dense(3, activation="softmax", name="task_stance_specific")(IA_stance)
    
#### topic #######
input_ids2 = Input(shape=(None,), dtype='int32', name='input_ids2')
attention_mask2 = Input(shape=(None,), dtype='int32', name='attention_mask2')
outputs = bertweet_model(input_ids=input_ids2, attention_mask=attention_mask2)
sequence_output = outputs.pooler_output
Q_s= Dense(100, activation="relu")(sequence_output)
K_s= Dense(100, activation="relu")(sequence_output)
V_s= Dense(100, activation="relu")(sequence_output)
IA_topic=Lambda(attentionScores)([Q_s,K_s,V_s])
IA_topic_flatten=Flatten()(IA_topic)
topic_specific_output=Dense(5, activation="softmax", name="task_topic_specific")(IA_topic)

#### hate #######
input_ids3 = Input(shape=(None,), dtype='int32', name='input_ids3')
attention_mask3 = Input(shape=(None,), dtype='int32', name='attention_mask3')
outputs = bertweet_model(input_ids=input_ids3, attention_mask=attention_mask3)
sequence_output = outputs.pooler_output
Q_s= Dense(100, activation="relu")(sequence_output)
K_s= Dense(100, activation="relu")(sequence_output)
V_s= Dense(100, activation="relu")(sequence_output)
IA_hate=Lambda(attentionScores)([Q_s,K_s,V_s])
IA_hate_flatten=Flatten()(IA_hate)
hate_specific_output=Dense(3, activation="softmax", name="task_hate_specific")(IA_hate)

#### sentiment #######
input_ids4 = Input(shape=(None,), dtype='int32', name='input_ids4')
attention_mask4 = Input(shape=(None,), dtype='int32', name='attention_mask4')
outputs = bertweet_model(input_ids=input_ids4, attention_mask=attention_mask4)
sequence_output = outputs.pooler_output
Q_s= Dense(100, activation="relu")(sequence_output)
K_s= Dense(100, activation="relu")(sequence_output)
V_s= Dense(100, activation="relu")(sequence_output)
IA_sent=Lambda(attentionScores)([Q_s,K_s,V_s])
IA_sent_flatten=Flatten()(IA_sent)
sent_specific_output=Dense(3, activation="softmax", name="task_sent_specific")(IA_sent)

####### correlated features #########
hate_stance=Lambda(correlatedFeatures)([IA_hate,IA_stance])
hate_stance= Dense(100, activation="relu")(hate_stance)

hate_topic=Lambda(correlatedFeatures)([IA_hate,IA_topic])
hate_topic= Dense(100, activation="relu")(hate_topic)

sent_stance=Lambda(correlatedFeatures)([IA_sent,IA_stance])
sent_stance= Dense(100, activation="relu")(sent_stance)

sent_topic=Lambda(correlatedFeatures)([IA_sent,IA_topic])
sent_topic= Dense(100, activation="relu")(sent_topic)

####### shared output #####
shared_input=Average()([hate_stance,hate_topic,sent_stance,sent_topic])
shared_output= Dense(100, activation="relu")(shared_input)

stance_shared_output=Dense(3, activation="softmax", name="task_stance_shared")(shared_output)
topic_shared_output=Dense(5, activation="softmax", name="task_topic_shared")(shared_output)
hate_shared_output=Dense(3, activation="softmax", name="task_hate_shared")(shared_output)
sent_shared_output=Dense(3, activation="softmax", name="task_sent_shared")(shared_output)

adv_stance=Lambda(gradRev)(stance_shared_output)
adv_topic=Lambda(gradRev)(topic_shared_output)
adv_hate=Lambda(gradRev)(hate_shared_output)
adv_sent=Lambda(gradRev)(sent_shared_output)

stance_adv_output=Dense(3, activation="softmax", name="task_adv_stance")(concatenate([stance_shared_output,adv_stance,stance_specific_output]))
topic_adv_output=Dense(5, activation="softmax", name="task_adv_topic")(concatenate([topic_shared_output,adv_topic,topic_specific_output]))
hate_adv_output=Dense(3, activation="softmax", name="task_adv_hate")(concatenate([hate_shared_output,adv_hate,hate_specific_output]))
sent_adv_output=Dense(3, activation="softmax", name="task_adv_sent")(concatenate([sent_shared_output,adv_sent,sent_specific_output]))

# Define the model
model=Model(inputs=[input_ids1, attention_mask1,input_ids2, attention_mask2,input_ids3, attention_mask3,input_ids4, attention_mask4],outputs=[stance_specific_output,topic_specific_output,hate_specific_output,sent_specific_output,stance_shared_output,topic_shared_output,hate_shared_output,sent_shared_output,stance_adv_output,topic_adv_output,hate_adv_output,sent_adv_output])

##Compile
model.compile(optimizer=Adam(0.0001),loss={'task_stance_specific':'categorical_crossentropy','task_topic_specific':'categorical_crossentropy',
    'task_hate_specific':'categorical_crossentropy','task_sent_specific':'categorical_crossentropy',
    'task_stance_shared':'categorical_crossentropy','task_topic_shared':'categorical_crossentropy',
    'task_hate_shared':'categorical_crossentropy','task_sent_shared':'categorical_crossentropy',
    'task_adv_stance':'categorical_crossentropy','task_adv_topic':'categorical_crossentropy',
    'task_adv_hate':'categorical_crossentropy','task_adv_sent':'categorical_crossentropy'},
    loss_weights={'task_stance_specific':1.0,'task_topic_specific':1.0,'task_hate_specific':0.4,'task_sent_specific':0.5,'task_stance_shared':1.0,'task_topic_shared':1.0,'task_hate_shared':0.4,'task_sent_shared':0.5,'task_adv_stance':1.0,'task_adv_topic':1.0,'task_adv_hate':0.4,'task_adv_sent':0.5}, metrics=['accuracy'])    
print(model.summary())

train_stance=to_categorical(train_stance_enc)
train_topic=to_categorical(train_topic_enc)
train_hate=to_categorical(train_hate_enc)
train_sent=to_categorical(train_sent_enc)
val_stance=to_categorical(val_stance_enc)
val_topic=to_categorical(val_topic_enc)
val_hate=to_categorical(val_hate_enc)
val_sent=to_categorical(val_sent_enc)

batch_size=32

model.fit([train_encodings['input_ids'], train_encodings['attention_mask'],train_encodings['input_ids'], train_encodings['attention_mask'],train_encodings['input_ids'], train_encodings['attention_mask'],train_encodings['input_ids'], train_encodings['attention_mask']],[train_stance,train_topic,train_hate,train_sent,train_stance,train_topic,train_hate,train_sent,train_stance,train_topic,train_hate,train_sent],steps_per_epoch=int(len(train_stance_enc) / batch_size),validation_data=([val_encodings['input_ids'], val_encodings['attention_mask'],val_encodings['input_ids'], val_encodings['attention_mask'],val_encodings['input_ids'], val_encodings['attention_mask'],val_encodings['input_ids'], val_encodings['attention_mask']],[val_stance,val_topic,val_hate,val_sent,val_stance,val_topic,val_hate,val_sent,val_stance,val_topic,val_hate,val_sent]),epochs=10,verbose=2)
predicted = model.predict([test_encodings['input_ids'], test_encodings['attention_mask'],test_encodings['input_ids'], test_encodings['attention_mask'],test_encodings['input_ids'], test_encodings['attention_mask'],test_encodings['input_ids'], test_encodings['attention_mask']])
print(predicted)

stance_specific_pred=predicted[0]
stance_shared_pred=predicted[4]
stance_adv_pred= 1-predicted[8]
result_=np.mean([stance_specific_pred,stance_shared_pred,stance_adv_pred],axis=0)
p_1 = np.argmax(result_, axis=1)
test_accuracy=accuracy_score(test_stance_enc, p_1)
print("test accuracy::::",test_accuracy)
target_names = ['-1.0','0.0','1.0']
class_rep=classification_report(test_stance_enc, p_1)
print("specific confusion matrix",confusion_matrix(test_stance_enc, p_1))
print(class_rep)
class_rep=classification_report(test_stance_enc, p_1, target_names=target_names,output_dict=True)
macro_avg=class_rep['macro avg']['f1-score']
macro_prec=class_rep['macro avg']['precision']
macro_rec=class_rep['macro avg']['recall']
print("macro f1 score",macro_avg)
print("macro_prec",macro_prec)
print("macro_rec",macro_rec)


### topic
topic_specific_pred=predicted[1]
topic_shared_pred=predicted[5]
topic_adv_pred=1- predicted[9]
result_=np.mean([topic_specific_pred,topic_shared_pred,topic_adv_pred],axis=0)
p_1 = np.argmax(result_, axis=1)
test_accuracy=accuracy_score(test_topic_enc, p_1)
print("test accuracy::::",test_accuracy)
target_names = ['1','2','3','4','5']
class_rep=classification_report(test_topic_enc, p_1)
print("specific confusion matrix",confusion_matrix(test_topic_enc, p_1))
print(class_rep)
class_rep=classification_report(test_topic_enc, p_1, target_names=target_names,output_dict=True)
macro_avg=class_rep['macro avg']['f1-score']
macro_prec=class_rep['macro avg']['precision']
macro_rec=class_rep['macro avg']['recall']
print("macro f1 score",macro_avg)
print("macro_prec",macro_prec)
print("macro_rec",macro_rec)
