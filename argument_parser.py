import argparse


def get_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "training_dir", help="input directory where all training images are found"
    )
    parser.add_argument(
        "test_dir", help="input directory where all test images are found"
    )
    parser.add_argument(
        "--model_type",
        help="Which base model to transfer learning from",
        default="MobileNetV2",
        choices=["VGG19", "MobileNetV2", "InceptionV3"],
    )
    parser.add_argument(
        "--skip_saving_model",
        help="Whether to skip saving the model. The model is saved by default after training",
        action="store_false",
    )
    parser.add_argument(
        "--load_model",
        help="Load from an existing model",
        action="store_true",
    )
    parser.add_argument(
        "--model_path",
        help="Which folder the trained models should be saved",
        default="./saved_models",
    )
    parser.add_argument(
        "--learning_rate",
        help="Base learning rate for optimizers",
        default=0.001,
        type=float,
    )
    # parser.add_argument(
    #     "--dropout_rate",
    #     help="Dropout rate for the dropout layers",
    #     default=0.2,
    #     type=float,
    # )
    # parser.add_argument(
    #     "--with_dropout",
    #     help="Determines if we are going to use a dropout layer in the models",
    #     action="store_true",
    # )
    parser.add_argument(
        "--epochs", help="Number of epochs to run the model for", default=3, type=int
    )
    parser.add_argument(
        "--batch_size", help="Batch size for the dataset", default=16, type=int
    )

    return parser.parse_args()
