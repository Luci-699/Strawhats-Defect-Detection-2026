import argparse
import yaml
from ultralytics import YOLO

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    model_name = config['model']['backbone'] + '.pt'
    if args.resume:
        model = YOLO(config['paths']['baseline_weights'])
    else:
        model = YOLO(model_name)
    
    epochs = args.epochs if args.epochs else config['training']['epochs_baseline']
    lr = args.lr if args.lr else float(config['training']['learning_rate'])
    run_name = args.name if args.name else 'baseline'
    
    model.train(
        data=args.data if args.data else config['paths']['dataset_yaml'],
        epochs=epochs,
        imgsz=args.imgsz if args.imgsz else config['training']['image_size'],
        batch=args.batch if args.batch else config['training']['batch_size'],
        device=args.device,
        optimizer=config['training']['optimizer'],
        lr0=float(lr),
        weight_decay=float(config['training']['weight_decay']),
        cos_lr=True,
        patience=20,
        augment=True,
        mosaic=1.0,
        flipud=0.5,
        degrees=10.0,
        project='runs',
        name=run_name,
        exist_ok=True
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train baseline YOLOv10 model.")
    parser.add_argument('--config', type=str, default='training/config.yaml', help='Path to config file.')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset.yaml.')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs.')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu or cuda).')
    parser.add_argument('--batch', type=int, default=None, help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=None, help='Image size.')
    parser.add_argument('--name', type=str, default=None, help='Run name (e.g. steel, aluminum, wood).')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate override.')
    parser.add_argument('--resume', action='store_true', help='Resume from best checkpoint.')
    args = parser.parse_args()
    main(args)
