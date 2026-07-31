import argparse
import os
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO

def main(args):
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    model_name = args.model if args.model else config['model']['backbone'] + '.pt'
    if args.resume:
        model = YOLO(config['paths']['baseline_weights'])
    else:
        model = YOLO(model_name)
    
    epochs = args.epochs if args.epochs else config['training']['epochs_baseline']
    lr = args.lr if args.lr else float(config['training']['learning_rate'])
    run_name = args.name if args.name else 'baseline'
    
    # Auto-clean any stale Ultralytics dataset cache files (*.cache)
    if sys.platform == 'win32':
        os.system('del /f /q /s data\\*.cache >nul 2>&1')
    for cache_file in Path("data").rglob("*.cache"):
        try:
            os.remove(cache_file)
            print(f"Removed stale cache: {cache_file}")
        except Exception:
            pass

    data_yaml_path = args.data if args.data else config['paths']['dataset_yaml']

    # CRITICAL FIX: Ultralytics resolves relative paths against its global settings.yaml
    # We must overwrite dataset.yaml to use the absolute path of the current project directory!
    with open(data_yaml_path, 'r') as f:
        ds_yaml = yaml.safe_load(f)
    
    if 'path' in ds_yaml:
        # We must reconstruct the absolute path from scratch using args.name
        # because previous runs may have corrupted dataset.yaml with a bad relative path.
        material_name = args.name
        resolved_path = (Path(data_yaml_path).resolve().parent.parent / "data" / "processed" / material_name).resolve()
        ds_yaml['path'] = str(resolved_path).replace('\\', '/')
        with open(data_yaml_path, 'w') as f:
            yaml.dump(ds_yaml, f, sort_keys=False)

    model.train(
        data=data_yaml_path,
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
        workers=4,
        cache='disk',
        exist_ok=True
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train baseline YOLOv10 model.")
    parser.add_argument('--config', type=str, default='training/config.yaml', help='Path to config file.')
    parser.add_argument('--data', type=str, default=None, help='Path to dataset.yaml.')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs.')
    parser.add_argument('--model', type=str, default=None, help='YOLO model variant (e.g. yolov10m.pt, yolov10l.pt).')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cpu or cuda).')
    parser.add_argument('--batch', type=int, default=None, help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=None, help='Image size.')
    parser.add_argument('--name', type=str, default=None, help='Run name (e.g. steel, aluminum, wood).')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate override.')
    parser.add_argument('--resume', action='store_true', help='Resume from best checkpoint.')
    args = parser.parse_args()
    main(args)
