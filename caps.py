def main():

    import numpy as np
    from keras.models import Sequential
    from keras.layers import Convolution2D
    from keras.layers import MaxPooling2D
    from keras.layers import Flatten
    from keras.layers import Dense, Dropout
    from keras import optimizers
    from sklearn.metrics import accuracy_score

    import tensorflow as tf
    import keras
    import keras.backend as K

    from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
    from keras import layers, models, optimizers
    from keras.preprocessing.image import ImageDataGenerator
    routings=2
    n_class=13
    basepath="E:/Brain Tumor Detection with Level new 1/Brain Tumor Detection with Level new 1/input_data_resized"
    input_shape = (64, 64, 3)
    classifier = Sequential()
    x = layers.Input(shape=input_shape)

   # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    classifier.add(Convolution2D(32, 1,  1, input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size =(2,2)))
  
    
  
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

   # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings,name='digitcaps')(primarycaps)

   # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
   # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)   
    classifier.add(Convolution2D(32, 1,  1, activation = 'relu'))
    # classifier.add(MaxPooling2D(pool_size =(2,2)))

    # classifier.add(Convolution2D(64, 1,  1, activation = 'relu'))
    # classifier.add(MaxPooling2D(pool_size =(2,2)))
    # classifier.add(Convolution2D(64, 1,  1, activation = 'relu'))
    # classifier.add(MaxPooling2D(pool_size =(2,2)))   

    classifier.add(Flatten())

   # Decoder network.
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y]) # The true label is used to mask the output of capsule layer. For training
    masked = Mask()(digitcaps) # Mask using the capsule with maximal length. For prediction

   # Shared Decoder model in training and prediction
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))

   # Models for training and evaluation (prediction)
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])    

    

    classifier.add(Dense(256, activation = 'relu'))
    classifier.add(Dropout(0.5))
    classifier.add(Dense(13, activation = 'softmax'))  #change class no.
    

    classifier.compile(
                  optimizer = optimizers.SGD(lr = 0.01),
                  loss = 'categorical_crossentropy',
                  metrics = ['accuracy'])
    

    from keras.preprocessing.image import ImageDataGenerator
    train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True)
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    training_set = train_datagen.flow_from_directory(
            basepath + '/training_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')
    
    test_set = test_datagen.flow_from_directory(
            basepath + '/test_set',
            target_size=(64, 64),
            batch_size=32,
            class_mode='categorical')
    
    steps_per_epoch = int( np.ceil(training_set.samples / 32) )
    val_steps = int( np.ceil(test_set.samples / 32) )
    
    model = classifier.fit_generator(
            training_set,
            steps_per_epoch=steps_per_epoch,
            epochs=3,
            validation_data = test_set,
            validation_steps =val_steps
          )
    
    #Saving the model
    #import h5py
    classifier.save(basepath + '/caps_network.h5')
    
    
    scores = classifier.evaluate(test_set, verbose=1)
    B="Testing Accuracy: %.2f%%" % (scores[1]*100)
    print(B)
    scores =  classifier.evaluate(training_set, verbose=1)
    C="Training Accuracy: %.2f%%" % (scores[1]*100)
    print(C)
    
    
    # msg=B+'\n'+ C 
    
    
    # import matplotlib.pyplot as plt
    # # summarize history for accuracy
    # plt.plot(model.history['accuracy'])
    # plt.plot(model.history['val_accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(basepath + "/accuracy.png",bbox_inches='tight')

    # plt.show()
    # # summarize history for loss
    
    # plt.plot(model.history['loss'])
    # plt.plot(model.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig(basepath + "/loss.png",bbox_inches='tight')
    
    # plt.show()

    
    # return msg
