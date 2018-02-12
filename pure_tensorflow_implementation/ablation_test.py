import train
import eval_auc
import tensorflow as tf

print "remove wind"
model = train.get_trained_model(tag='no_wind', remove_image = False, remove_wind = True, remove_hcad = False)
eval_auc.evaluate_auc(model)
tf.reset_default_graph()

print "remove image"
model = train.get_trained_model(tag='no_image', remove_image = True, remove_wind = False, remove_hcad = False)
eval_auc.evaluate_auc(model)
tf.reset_default_graph()

print "remove hcad"
model = train.get_trained_model(tag='no_hcad', remove_image = False, remove_wind = False, remove_hcad = True)
eval_auc.evaluate_auc(model)
tf.reset_default_graph()


print "include everything"
model = train.get_trained_model(tag='include_all_lr001_with_val', remove_image = False, remove_wind = False, remove_hcad = False)
eval_auc.evaluate_auc(model)
tf.reset_default_graph()

