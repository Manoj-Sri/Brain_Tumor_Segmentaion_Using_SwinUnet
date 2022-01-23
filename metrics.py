import tensorflow as tf
import tensorlayer as tl
import keras.backend as K

def dice_coef(y_true,y_pred,loss_type='jaccard',smooth=1.):

    y_true_f = tf.keras.layers.Flatten(y_true)
    y_pred_f = tf.keras.layers.Flatten(y_pred)


#     y_true_f = tl.layers.FlattenLayer(y_true)
#     y_pred_f = tl.layers.FlattenLayer(y_pred)

    intersection = tf.reduce_sum(y_true_f * y_pred_f)

    if loss_type == 'jaccard':
        union = tf.reduce_sum(tf.square(y_true_f)) + tf.reduce_sum(tf.square(y_pred_f))
    elif loss_type == 'sorenson':
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    else:
        raise ValueError("Unkown loss type : "+loss_type)

    return (2.*intersection + smooth)/(union + smooth)



def dice_coefficient(y_true, y_pred, smooth=1):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
    dice = tf.reduce_mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


def dice_loss(y_true,y_pred,loss_type='jaccard',smooth=1):
#     return 1-dice_coef(y_true,y_pred,loss_type=loss_type,smooth=smooth)
    return 1-dice_coefficient(y_true,y_pred,smooth=smooth)


def mean_iou(y_true,y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    
    union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])

    return intersection/(union-intersection)




def _specificity(y_true,y_pred):
   
    neg_y_true = 1 - y_true
    neg_y_pred = 1 - y_pred
    fp = K.sum(neg_y_true * y_pred)
    tn = K.sum(neg_y_true * neg_y_pred)
    specificity = tn / (tn + fp + K.epsilon())
    return specificity


def sensitivity(y_true, y_pred):  #recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())





def specificity(y_true, y_pred):
    true_negatives = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = K.sum(K.round(K.clip(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + K.epsilon())
