if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from ultralytics import YOLO

    # Load YOLOv8 classification model
    model = YOLO('yolov8x-cls.pt')

    # Find and replace the classification head properly
    num_classes = 2

    for name, module in model.model.named_modules():
        if isinstance(module, nn.Linear):
            input_features = module.in_features  # Get input size of the last FC layer

            setattr(
                model.model,
                name,
                nn.Sequential(
                    nn.Linear(input_features, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(),
                    nn.Dropout(0.4),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128,32),                    
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(32, num_classes)
                )

            )
            break  # Stop after replacing the first linear layer found

    # Check if the modification was successful
    print(model.model)

    # Train the model
    dataset_path = r""
    model.train(
        data=dataset_path,
        epochs=10000,
        batch=16,  # Adjust if needed
        imgsz=224,
        project=r"project",
        name='yolo',
        optimizer='AdamW',
        lrf=0.001,
        lr0=0.0001,
        weight_decay=0.01,
        patience=1000,
    )

    # Validate after training
    val_results = model.val(
        data=dataset_path,
        split="val",
        batch=32,
        imgsz=224,
    )
    print("Validation Results:", val_results)

    # Test the model on an independent test dataset
    test_results = model.val(
        data=dataset_path,
        split="test",
        batch=32,
        imgsz=224,
    )
    print("Test Results:", test_results)
