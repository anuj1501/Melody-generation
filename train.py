import tensorflow.keras as keras
from preprocess import generating_training_sequences,SEQUENCE_LENGTH

OUTPUT_UNITS = 38
NUM_UNITS = [256]
LOSS = "sparse_categorical_crossentropy"
LEARNING_RATE = 0.001
EPOCHS = 50
BATCH_SIZE = 64 


def build_model(output_units,num_units,loss,learning_rate):

    #create the architecture

    input_data = keras.layers.Input(shape=(None,output_units))
    x = keras.layers.LSTM(num_units[0])(input_data)
    x = keras.layers.Dropout(0.2)(x)

    output = keras.layers.Dense(output_units,activation="softmax")(x)

    model = keras.Model(input_data,output)

    #compile the model
    model.compile(loss = loss,optimizer=keras.optimizers.Adam(lr=learning_rate),metrics=["accuracy"])

    print(model.summary())

    return model


def train(output_units,num_units,loss,learning_rate):

    #generate the training sequences
    inputs,targets = generating_training_sequences(SEQUENCE_LENGTH)

    #build the model
    model = build_model(output_units,num_units,loss,learning_rate)

    #train the model
    model.fit(inputs,targets,epochs=EPOCHS,batch_size = BATCH_SIZE)

    #save the model
    model.save("model.h5")


if __name__ == "__main__":

    train()