from keras.models import model_from_json
import tensorflow as tf
def init():
  json_file = open(r'C:\Users\Kedar\Google Drive\Google Drive\Learning\Data Science\Projects\Deep Learning\MNIST Deploy\model.json','r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  #load weights into new model
  loaded_model.load_weights(r"C:\Users\Kedar\Google Drive\Google Drive\Learning\Data Science\Projects\Deep Learning\MNIST Deploy\model.h5")
  print("Loaded Model from disk")
  #compile and evaluate loaded model
  loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  global graph
  graph = tf.compat.v1.get_default_graph()
  return loaded_model,graph