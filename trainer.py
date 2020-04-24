import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, LearningRateScheduler
from tensorflow.keras.optimizers import SGD, Adam
import helpers
import augmentation
from ssd_loss import CustomLoss
from training_utils import generator
from bbox_utils import generate_prior_boxes

args = helpers.handle_args()
if args.handle_gpu:
    helpers.handle_gpu_compatibility()

batch_size = 8
epochs = 150
load_weights = False
with_voc2012 = True
backbone = args.backbone
#
assert backbone in ["mobilenet_v2", "vgg16"]
if backbone == "mobilenet_v2":
    from models.ssd_mobilenet_v2 import get_model, init_model
else:
    from models.ssd_vgg16 import get_model, init_model
#
hyper_params = helpers.get_hyper_params(backbone)
#
VOC_train_data, VOC_info = helpers.get_dataset("voc/2007", "train+validation")
VOC_val_data, _ = helpers.get_dataset("voc/2007", "test")
VOC_train_total_items = helpers.get_total_item_size(VOC_info, "train+validation")
VOC_val_total_items = helpers.get_total_item_size(VOC_info, "test")

if with_voc2012:
    VOC_2012_train_data, VOC_2012_info = helpers.get_dataset("voc/2012", "train+validation")
    VOC_2012_train_total_items = helpers.get_total_item_size(VOC_2012_info, "train+validation")
    VOC_train_total_items += VOC_2012_train_total_items
    VOC_train_data = VOC_train_data.concatenate(VOC_2012_train_data)

step_size_train = helpers.get_step_size(VOC_train_total_items, batch_size)
step_size_val = helpers.get_step_size(VOC_val_total_items, batch_size)
labels = helpers.get_labels(VOC_info)
# We add 1 class for background
hyper_params["total_labels"] = len(labels) + 1
img_size = hyper_params["img_size"]

VOC_train_data = VOC_train_data.map(lambda x : helpers.preprocessing(x, img_size, img_size, augmentation.apply))
VOC_val_data = VOC_val_data.map(lambda x : helpers.preprocessing(x, img_size, img_size))

padded_shapes, padding_values = helpers.get_padded_batch_params()
VOC_train_data = VOC_train_data.shuffle(batch_size*4).padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
VOC_val_data = VOC_val_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)
#
ssd_model = get_model(hyper_params)
ssd_custom_losses = CustomLoss(hyper_params["neg_pos_ratio"], hyper_params["loc_loss_alpha"])
ssd_model.compile(optimizer=Adam(learning_rate=1e-3),
                  loss=[ssd_custom_losses.loc_loss_fn, ssd_custom_losses.conf_loss_fn])
init_model(ssd_model)
#
ssd_model_path = helpers.get_model_path(backbone)
if load_weights:
    ssd_model.load_weights(ssd_model_path)
ssd_log_path = helpers.get_log_path(backbone)
# We calculate prior boxes for one time and use it for all operations because of the all images are the same sizes
prior_boxes = generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
ssd_train_feed = generator(VOC_train_data, prior_boxes, hyper_params)
ssd_val_feed = generator(VOC_val_data, prior_boxes, hyper_params)

checkpoint_callback = ModelCheckpoint(ssd_model_path, monitor="val_loss", save_best_only=True, save_weights_only=True)
tensorboard_callback = TensorBoard(log_dir=ssd_log_path)
learning_rate_callback = LearningRateScheduler(helpers.scheduler, verbose=0)

ssd_model.fit(ssd_train_feed,
              steps_per_epoch=step_size_train,
              validation_data=ssd_val_feed,
              validation_steps=step_size_val,
              epochs=epochs,
              callbacks=[checkpoint_callback, tensorboard_callback, learning_rate_callback])
