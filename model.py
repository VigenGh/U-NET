
imgs=os.listdir(r"C:/Users/USER/Desktop/segm/images")[:600]
anot=os.listdir(r"C:/Users/USER/Desktop/segm/masks")[:600]
train_img=[]
train_anot=[]
arr=np.ones((30,30))

for u in imgs:
    img1=r"C:/Users/USER/Desktop/segm/images/"+u
    img1=img.load_img(img1)
    img1=tf.image.resize(img1,[224,224])
    img1=img.img_to_array(img1)
    train_img.append(img1)

for y in anot:
    k=np.ones((224,224))
    img2 = r"C:/Users/USER/Desktop/segm/masks/" + y
    img2 = img.load_img(img2)
    img2 = tf.image.resize(img2, [224, 224])
    img2 = img.img_to_array(img2)
    img2 = tf.image.rgb_to_grayscale(img2)
    z=(np.array(img2)[:,:,0]>0)*k
    train_anot.append(z)
train_img=np.array(train_img,dtype=int)
train_anot=np.array(train_anot,dtype=int)
print(train_img[0].shape)

x_train,x_test,y_train,y_test=train_test_split(train_img,train_anot,test_size=0.25,random_state=1)

input = tf.keras.Input(shape=(224,224,3))
c1 = tf.keras.layers.Conv2D(32,kernel_size=3,padding="same",activation="relu")(input)
c2 = tf.keras.layers.Conv2D(32,kernel_size=3,padding="same",activation="relu")(c1)
c3 = tf.keras.layers.MaxPool2D((2,2),strides=(2,2))(c2)
c4 = tf.keras.layers.Conv2D(64,kernel_size=3,padding="same",activation="relu")(c3)
c5 = tf.keras.layers.Conv2D(64,kernel_size=3,padding="same",activation="relu")(c4)
c6 = tf.keras.layers.MaxPool2D((2,2),strides=(2,2))(c5)
c7 = tf.keras.layers.Conv2D(128,kernel_size=3,padding="same",activation="relu")(c6)
c8 = tf.keras.layers.Conv2D(128,kernel_size=3,padding="same",activation="relu")(c7)
c9 = tf.keras.layers.MaxPool2D((2,2),strides=(2,2))(c8)
c10 = tf.keras.layers.Conv2D(256,kernel_size=3,padding="same",activation="relu")(c9)
c11 = tf.keras.layers.Conv2D(256,kernel_size=3,padding="same",activation="relu")(c10)
t0=tf.keras.layers.Conv2DTranspose(128,kernel_size=2,strides=(2,2),activation="relu")(c11)
c12=tf.keras.layers.concatenate([t0,c8],axis=-1)
c13 = tf.keras.layers.Conv2D(128,kernel_size=3,padding="same",activation="relu")(c12)
c14 = tf.keras.layers.Conv2D(128,kernel_size=3,padding="same",activation="relu")(c13)
t1=tf.keras.layers.Conv2DTranspose(128,kernel_size=2,strides=(2,2),activation="relu")(c14)
c15=tf.keras.layers.concatenate([t1,c5],axis=-1)
c16 = tf.keras.layers.Conv2D(64,kernel_size=3,padding="same",activation="relu")(c15)
c17 = tf.keras.layers.Conv2D(64,kernel_size=3,padding="same",activation="relu")(c16)
t2=tf.keras.layers.Conv2DTranspose(32,kernel_size=2,strides=(2,2),activation="relu")(c17)
c18=tf.keras.layers.concatenate([t2,c2],axis=-1)
c19 = tf.keras.layers.Conv2D(32,kernel_size=3,padding="same",activation="relu")(c18)
c20 = tf.keras.layers.Conv2D(32,kernel_size=3,padding="same",activation="relu")(c19)
output = tf.keras.layers.Conv2D(1,kernel_size=1,activation="sigmoid")(c20)
model=tf.keras.Model(inputs=input,outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=tf.keras.losses.BinaryCrossentropy(),metrics=["accuracy"])
model.fit(x_train,y_train,epochs=15,batch_size=32)
model.save("model2.h5")

model=load_model("model2.h5")
prediction=model.predict(train_img[:4])

o=0
for t in range(8):
    p = np.ones((224, 224))
    if o<4:
        plt.subplot(2,4,t+1)
        plt.imshow(p*(prediction[t,:,:,0]>0.5))
    else:
        plt.subplot(2, 4, t + 1)
        plt.imshow(train_anot[t-4, :, :])
    o+=1
