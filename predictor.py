import tensorflow as tf
from utils import bbox_utils, data_utils, drawing_utils, io_utils, train_utils
from models.decoder import get_decoder_model

args = io_utils.handle_args()
if args.handle_gpu:
    io_utils.handle_gpu_compatibility()

batch_size = 1
use_custom_images = False
custom_image_path = "data/images/"
backbone = args.backbone
io_utils.is_valid_backbone(backbone)
#
if backbone == "mobilenet_v2":
    from models.ssd_mobilenet_v2 import get_model, init_model
else:
    from models.ssd_vgg16 import get_model, init_model
#
hyper_params = train_utils.get_hyper_params(backbone)
#
test_data, info = data_utils.get_dataset("voc/2007", "test")
labels = data_utils.get_labels(info)
labels = ["bg"] + labels
hyper_params["total_labels"] = len(labels)
img_size = hyper_params["img_size"]

if use_custom_images:
    test_data = data_utils.get_image_data_from_folder(custom_image_path, img_size, img_size)
else:
    test_data = test_data.map(lambda x : data_utils.preprocessing(x, img_size, img_size))
    padded_shapes, padding_values = data_utils.get_padded_batch_params()
    test_data = test_data.padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)

ssd_model = get_model(hyper_params)
ssd_model_path = io_utils.get_model_path(backbone)
ssd_model.load_weights(ssd_model_path)

prior_boxes = bbox_utils.generate_prior_boxes(hyper_params["feature_map_shapes"], hyper_params["aspect_ratios"])
ssd_decoder_model = get_decoder_model(ssd_model, prior_boxes, hyper_params)

for image_data in test_data:
    imgs, _, _ = image_data
    pred_bboxes, pred_labels, pred_scores = ssd_decoder_model.predict_on_batch(imgs)
    for i, img in enumerate(imgs):
        denormalized_bboxes = bbox_utils.denormalize_bboxes(pred_bboxes[i], img_size, img_size)
        drawing_utils.draw_bboxes_with_labels(img, denormalized_bboxes, pred_labels[i], pred_scores[i], labels)
